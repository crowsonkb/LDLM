#!/usr/bin/env python3

"""Trains a latent diffusion language model."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain, islice
import json
import math
from pathlib import Path
import random
import sys
import zipfile

import accelerate
from datasets import load_dataset
from einops import rearrange
import k_diffusion as K
import peft
import safetensors.torch as safetorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data, _pytree as pytree
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print = tqdm.external_write_mode()(print)


def cosine_warmup(steps, value=1.0):
    return lambda i: value * math.sin(min(i / steps, 1) * math.pi / 2) ** 2


class ZippedConversationsDataset:
    def __init__(self, zip_file):
        self.training_items = []
        zip_ = zipfile.ZipFile(zip_file)
        for file_ in zip_.namelist():
            if file_.endswith("/"):  # Skip directories
                continue
            if file_.startswith("__MACOSX"):  # Mac OS X adds garbage to zips
                continue
            with zip_.open(file_) as infile:
                if file_.endswith(".txt"):
                    self.training_items.append(infile.read())
                else:
                    conversation = json.load(infile)
                    for id_ in conversation["responseDict"]:
                        branch = conversation["responseDict"][id_]
                        if branch["rating"] == True:
                            text = branch["prompt"] + branch["text"]
                            self.training_items.append(text)
        random.shuffle(self.training_items)

    def __len__(self):
        return len(self.training_items)

    def __next__(self):
        return random.sample(self.training_items, 1)[0]


# LDLM VAE


@contextmanager
def set_adapter(model, adapter_name):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    finally:
        model.set_adapter(old_adapter_name)


def gumbel_like(x):
    return torch.rand_like(x).log_().nan_to_num_().neg_().log_().neg_()


@contextmanager
def disable_causal_mask():
    import transformers.models.llama.modeling_llama as modeling

    decoder_fn = modeling._make_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    try:
        modeling._make_causal_mask = encoder_fn
        yield
    finally:
        modeling._make_causal_mask = decoder_fn


class VAEComponent(nn.Module):
    def __init__(self, d_model, z_dim):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.f = nn.Linear(d_model, 1)
        self.w_e = nn.Linear(d_model, z_dim)
        self.w_d = nn.Linear(z_dim, d_model)
        nn.init.orthogonal_(self.w_e.weight)
        with torch.no_grad():
            self.w_d.weight.copy_(self.w_e.weight.T)

    def encode(self, hidden_states, attention_mask):
        scores = self.f(hidden_states)
        scores = scores + attention_mask[:, :, None].log().nan_to_num()
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights, dim=1)
        return self.w_e(pooled)

    def sample(self, mean, tau=1.0):
        return mean + torch.randn_like(mean) * tau**0.5

    def decode(self, z):
        return self.w_d(z)


class DecoderOnlyTransformerVAE(nn.Module):
    def __init__(self, model_name, device, z_dim=768, lora_rank=32, dropout=0.0):
        super().__init__()
        self.device = device
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        peft_config = peft.LoraConfig(
            peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=8,
            lora_dropout=dropout,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
        )
        self.model = peft.get_peft_model(model, peft_config, "encoder")
        self.model.add_adapter("decoder", peft_config)
        self.model.config.output_hidden_states = True
        self.vae = VAEComponent(self.model.config.hidden_size, z_dim).to(device)

    def save_pretrained(self, path):
        path = Path(path)
        self.model.save_pretrained(path, safe_serialization=True)
        safetorch.save_file(self.vae.state_dict(), path / "vae.safetensors")

    def load_pretrained(self, path, is_trainable=False):
        path = Path(path)
        self.model.delete_adapter("encoder")
        self.model.load_adapter(path / "encoder", "encoder", is_trainable=is_trainable)
        self.model.delete_adapter("decoder")
        self.model.load_adapter(path / "decoder", "decoder", is_trainable=is_trainable)
        self.vae.load_state_dict(safetorch.load_file(path / "vae.safetensors"))

    def encode(self, input_ids, attention_mask):
        with set_adapter(self.model, "encoder"), disable_causal_mask():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        return self.vae.encode(outputs.hidden_states[-1], attention_mask)

    def input_ids_to_embeds(self, input_ids):
        embed_weight = self.model.get_input_embeddings().weight
        input_one_hots = F.one_hot(input_ids, num_classes=self.model.config.vocab_size)
        return input_one_hots.to(embed_weight) @ embed_weight

    @torch.no_grad()
    def generate(self, z, input_ids, attention_mask, n_tokens, tau=1.0):
        z_embed = self.vae.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids)
        inputs_embeds = torch.cat([z_embed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask.new_ones([attention_mask.shape[0], 1]), attention_mask], dim=1
        )
        new_embeds, past = None, None
        with set_adapter(self.model, "decoder"):
            for _ in range(n_tokens):
                outputs = self.model(
                    inputs_embeds=inputs_embeds if past is None else new_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = outputs.logits[:, -1:, :].float()
                new_input_ids = torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)
                input_ids = torch.cat([input_ids, new_input_ids], dim=1)
                new_embeds = self.input_ids_to_embeds(new_input_ids)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
                )
                past = outputs.past_key_values
        return input_ids

    def forward(self, input_ids, attention_mask, decoder_prefix_ids, decoder_prefix_mask):
        input_ids_all = torch.cat([decoder_prefix_ids, input_ids], dim=1)
        attn_mask_all = torch.cat([decoder_prefix_mask, attention_mask], dim=1)
        mean = self.encode(input_ids, attention_mask)
        z = self.vae.sample(mean)
        z_embed = self.vae.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids_all)
        inputs_embeds = torch.cat([z_embed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask.new_ones([attn_mask_all.shape[0], 1]), attn_mask_all], dim=1
        )
        with set_adapter(self.model, "decoder"):
            outputs = self.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
            )
        return outputs, mean


# LDLM diffusion model


def padding_mask_to_attn_mask(padding_mask):
    n, s = padding_mask.shape
    eye = torch.eye(s, device=padding_mask.device)
    base_mask = torch.ones([n, s, s], device=padding_mask.device)
    mask = torch.maximum(base_mask * padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2), eye)
    return mask.bool()


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_model * 4)
        self.gate_proj = nn.Linear(d_model, d_model * 4)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x = self.norm(x)
        main = self.up_proj(x)
        gate = self.gate_proj(x)
        x = self.act(gate) * main
        x = self.down_proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.n_heads = d_model // 64
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask):
        x = self.norm(x)
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)
        q = rearrange(q, "n l (h e) -> n h l e", h=self.n_heads)
        k = rearrange(k, "n l (h e) -> n h l e", h=self.n_heads)
        v = rearrange(v, "n l (h e) -> n h l e", h=self.n_heads)
        attn_mask = padding_mask_to_attn_mask(padding_mask).unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "n h l e -> n l (h e)")
        x = self.o_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = SelfAttention(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, padding_mask):
        x = x + self.attn(x, padding_mask)
        x = x + self.ff(x)
        return x


class LDLM(nn.Module):
    def __init__(self, n_layers, d_model, z_dim, ctx_len):
        super().__init__()
        self.config = {"n_layers": n_layers, "d_model": d_model, "z_dim": z_dim, "ctx_len": ctx_len}
        seq_len = ctx_len + 2
        self.time_emb = K.layers.FourierFeatures(1, d_model)
        self.z_in_proj = nn.Linear(z_dim, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) / d_model**0.5)
        self.norm_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.z_out_proj = nn.Linear(d_model, z_dim)

    def forward(self, z, sigma, z_prev, padding_mask):
        z = self.z_in_proj(z[:, None])
        z_prev = self.z_in_proj(z_prev)
        time_emb = self.time_emb(sigma[:, None, None].log() / 4)
        x = torch.cat([time_emb, z, z_prev], dim=1)
        allow_attend = padding_mask.new_ones([z.shape[0], 1])
        padding_mask = torch.cat([allow_attend, allow_attend, padding_mask], dim=1)
        x = x + self.pos_emb[: x.shape[1]]
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.norm_out(x)
        z_out = self.z_out_proj(x[:, 1])
        return z_out


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def batch_to_tensors(batch, tokenizer, context, device="cpu"):
    document_tokens = tokenizer(batch).input_ids
    for tokens in document_tokens:
        tokens.append(tokenizer.eos_token_id)
    chunks = list(batched(chain.from_iterable(document_tokens), context))
    seq_len = max(len(x) for x in chunks)
    input_ids = torch.zeros(len(chunks), seq_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(chunks), seq_len, dtype=torch.long, device=device)
    for i, x in enumerate(chunks):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
        attention_mask[i, : len(x)] = 1
    return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4, help="microbatch size")
    parser.add_argument(
        "--pretraining-dataset",
        default="togethercomputer/RedPajama-Data-1T-Sample",
        help="bulk pretraining dataset to tune on",
    )
    # parser.add_argument("--user-dataset", type=Path, help="MiniHF user dataset path")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="use gradient checkpointing",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--model",
        type=str,
        default="openlm-research/open_llama_3b_v2",
        help="model name",
    )
    parser.add_argument("--context", type=int, default=64, help="context window length")
    parser.add_argument("--output", type=Path, required=True, help="path to save adapter")
    parser.add_argument("--rank", type=int, default=32, help="the lora rank")
    parser.add_argument("--save-every", type=int, default=1000, help="save every n steps")
    # parser.add_argument("--start-from", type=str, help="start from existing lora")
    parser.add_argument("--z-dim", type=int, default=768, help="the latent dimension")
    parser.add_argument("--vae", type=Path, required=True, help="the vae checkpoint to use")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device if accelerator.num_processes > 1 else "cuda:0"
    is_main = accelerator.is_main_process
    print0 = accelerator.on_main_process(print)

    if Path(args.model).exists():
        model_name = Path(args.model).resolve()
    else:
        model_name = args.model

    print0(f"Loading model: {model_name}", file=sys.stderr)
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        model_vae = DecoderOnlyTransformerVAE(
            model_name, device, z_dim=args.z_dim, lora_rank=args.rank, dropout=args.dropout
        )

    model_vae.load_pretrained(args.vae)
    n_layers = 12
    d_model = 1024
    model = LDLM(n_layers=n_layers, d_model=d_model, z_dim=args.z_dim, ctx_len=1)
    model_ema = deepcopy(model)
    ema_sched = K.utils.EMAWarmup(power=2 / 3, max_value=0.9999)
    accelerator.wait_for_everyone()
    print0("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    input_ids_all, attention_mask_all = [], []
    for i in range(10):
        data_file = safetorch.load_file(f"shard_{i}.safetensors")
        input_ids = torch.split(data_file["input_ids"], args.context * 2, dim=1)
        attention_mask = torch.split(data_file["attention_mask"], args.context * 2, dim=1)
        if input_ids[-1].shape[1] != args.context * 2:
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
        input_ids_all.extend(input_ids)
        attention_mask_all.extend(attention_mask)
    del data_file, input_ids, attention_mask
    input_ids_all = torch.cat(input_ids_all)
    attention_mask_all = torch.cat(attention_mask_all)
    valid_indices = attention_mask_all.sum(dim=1) > args.context
    input_ids_all = input_ids_all[valid_indices]
    attention_mask_all = attention_mask_all[valid_indices]
    del valid_indices

    preprocessed = data.TensorDataset(input_ids_all, attention_mask_all)

    dataloader = data.DataLoader(
        preprocessed,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model, model_ema, opt, dataloader = accelerator.prepare(model, model_ema, opt, dataloader)
    model = K.layers.SimpleLossDenoiser(model)
    model_ema = K.layers.SimpleLossDenoiser(model_ema)

    i = 0
    measure_ae_scale = False
    ae_scale_sum = torch.tensor(0.0, device=device)
    ae_scale = 1.527548  # TODO: don't hardcode this

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def demo(model, input_ids, attention_mask, n_tokens):
        bs = 4
        tau = 0.1
        sigma_min, sigma_max = 0.01, 100
        sigmas = K.sampling.get_sigmas_karras(100, sigma_min, sigma_max, device=device)
        input_ids = input_ids[:bs, :n_tokens]
        attention_mask = attention_mask[:bs, :n_tokens]
        x = torch.randn([bs, args.z_dim], device=device) * sigma_max
        z_prev = model_vae.encode(input_ids, attention_mask)[:, None] / ae_scale
        extra_args = {
            "z_prev": z_prev,
            "padding_mask": torch.ones([bs, 1], dtype=torch.long, device=device),
        }
        mean = K.sampling.sample_dpmpp_2m_sde(
            model, x, sigmas, extra_args=extra_args, disable=not is_main
        )
        z = model_vae.vae.sample(mean * ae_scale, tau=tau)
        output_ids = model_vae.generate(z, input_ids, attention_mask, n_tokens, tau=tau)
        out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in output_ids]
        print0("======")
        for i, out_text in enumerate(out_texts):
            print0(out_text)
            if i < len(out_texts) - 1:
                print0("===")
        print0("======")

    def save():
        print0(f"### Saving model to {args.output}", file=sys.stderr)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model_ema.inner_model)
            args.output.mkdir(exist_ok=True, parents=True)
            obj = pytree.tree_map(lambda x: x.half(), unwrapped_model.state_dict())
            config_obj = dict(config=unwrapped_model.config, ae_scale=ae_scale)
            state_obj = {"step": i}
            safetorch.save_file(obj, args.output / "model.safetensors")
            with open(args.output / "config.json", "w") as f:
                json.dump(config_obj, f)
            with open(args.output / "state.json", "w") as f:
                json.dump(state_obj, f)

    accelerator.wait_for_everyone()
    for epoch in trange(args.epochs, disable=not is_main):
        for input_ids, attention_mask in tqdm(dataloader, disable=not is_main):
            input_ids = input_ids.long()

            if i % 100 == 0:
                demo(model_ema, input_ids, attention_mask, args.context)

            with accelerator.accumulate(model), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                input_ids_1, input_ids_2 = input_ids.chunk(2, dim=1)
                attn_mask_1, attn_mask_2 = attention_mask.chunk(2, dim=1)

                padding_mask = torch.rand([input_ids_1.shape[0], 1], device=device).gt(0.2).long()
                z_prev = model_vae.encode(input_ids_1, attn_mask_1)[:, None] / ae_scale
                z = model_vae.encode(input_ids_2, attn_mask_2) / ae_scale

                if measure_ae_scale:
                    # TODO: measure AE scale in the AE training code instead
                    ae_mean_sq = (z.detach().pow(2).mean() + z_prev.detach().pow(2).mean()) / 2
                    ae_mean_sq = accelerator.reduce(ae_mean_sq, "mean")
                    ae_scale_sum += ae_mean_sq
                    ae_scale_avg = torch.sqrt(ae_scale_sum / (i + 1))
                    print0("ae_scale_avg", ae_scale_avg.item(), file=sys.stderr)
                    if i == 100:
                        return

                noise = torch.randn_like(z)
                sigma = K.utils.rand_v_diffusion(
                    z.shape[:1], min_value=1e-3, max_value=1e3, device=device
                )
                losses = model.loss(z, noise, sigma, z_prev=z_prev, padding_mask=padding_mask)
                loss = losses.mean()

                accelerator.backward(losses.mean())
                opt.step()
                opt.zero_grad()
                # TODO: only update the EMA model after all grad accumulation steps are done
                K.utils.ema_update(model, model_ema, ema_sched.get_value())
                ema_sched.step()

                loss_global = accelerator.reduce(loss, "mean")
                print0(f"epoch: {epoch}, step: {i}, loss: {loss_global.item():g}", file=sys.stderr)
                i += 1

                if i % args.save_every == 0:
                    save()

        save()


if __name__ == "__main__":
    main()
