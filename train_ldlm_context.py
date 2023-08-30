#!/usr/bin/env python3

"""Trains a latent diffusion language model."""

import argparse
from copy import deepcopy
from itertools import chain, islice
import json
import math
import os
from pathlib import Path
import random
import sys

import accelerate
from einops import rearrange
import k_diffusion as K
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import optim
from torch.nn import functional as F
from torch.utils import data, _pytree as pytree
from tqdm import trange, tqdm
from transformers import AutoTokenizer

from ldlm_model import LDLM, X0Denoiser
from vae_model import DecoderOnlyTransformerVAE


print = tqdm.external_write_mode()(print)


def cosine_warmup(steps, value=1.0):
    return lambda i: value * math.sin(min(i / steps, 1) * math.pi / 2) ** 2


def ema_update_dict(values, updates, decay):
    for k, v in updates.items():
        if k not in values:
            values[k] = v
        else:
            values[k] *= decay
            values[k] += (1 - decay) * v
    return values


class PreprocessedSlimPajamaLoader(data.IterableDataset):
    def __init__(self, dataset_root):
        self.data = (), ()
        self.index = []
        self.shard_paths = []
        dir_stack = [dataset_root]
        while dir_stack:
            current_dir = dir_stack.pop()
            for entry in os.scandir(current_dir):
                if entry.is_dir():
                    dir_stack.append(entry.path)
                elif entry.path.endswith(".safetensors"):
                    self.shard_paths.append(entry.path)
        random.shuffle(self.shard_paths)

    def _refresh(self):
        shards = []
        for i in range(100):
            try:
                shard_path = self.shard_paths.pop()
                shards.append(safetorch.load_file(shard_path))
            except IndexError:
                break
        input_ids = torch.cat([x["input_ids"] for x in shards])
        attention_mask = torch.cat([x["attention_mask"] for x in shards])
        return input_ids, attention_mask

    def __iter__(self):
        return self

    def __next__(self):
        try:
            i = self.index.pop()
        except IndexError:
            self.data = self._refresh()
            if not self.data[0].numel():
                raise StopIteration
            self.index = list(torch.randperm(len(self.data[0])))
            i = self.index.pop()
        return self.data[0][i], self.data[1][i]


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
    parser.add_argument("--preprocessed", type=Path, required=True, help="preprocessed dataset dir")
    parser.add_argument("--batch-size", type=int, default=4, help="microbatch size")
    parser.add_argument(
        "--pretraining-dataset",
        default="togethercomputer/RedPajama-Data-1T-Sample",
        help="bulk pretraining dataset to tune on",
    )
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

    torch.backends.cuda.matmul.allow_tf32 = True
    # this try block can be removed once pytorch 2.1 is released
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

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

    ctx_len = 40
    n_layers = 12
    d_model = 1024
    ae_scale = 1.56105  # TODO: don't hardcode this
    demo_every = 1000
    model = LDLM(n_layers=n_layers, d_model=d_model, z_dim=args.z_dim, ctx_len=ctx_len - 1, sigma_data=ae_scale)
    model_ema = deepcopy(model)
    ema_sched = K.utils.EMAWarmup(power=0.7, max_value=0.9999)
    ema_stats = {}
    accelerator.wait_for_everyone()
    print0("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    groups = model.param_groups(base_lr=args.lr)
    opt = optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-3)

    preprocessed = PreprocessedSlimPajamaLoader(args.preprocessed)

    dataloader = data.DataLoader(
        preprocessed,
        batch_size=args.batch_size,
        drop_last=True,
    )

    model, model_ema, opt, dataloader = accelerator.prepare(model, model_ema, opt, dataloader)

    gns_stats = None
    if accelerator.num_processes > 1:
        gns_stats_hook = K.gns.DDPGradientStatsHook(model)
        gns_stats = K.gns.GradientNoiseScale()

    model = X0Denoiser(model, sigma_data=ae_scale)
    model_ema = X0Denoiser(model_ema, sigma_data=ae_scale)

    i = 0
    measure_ae_scale = False
    ae_scale_sum = torch.tensor(0.0, device=device)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def demo(model, input_ids, attention_mask, n_tokens):
        bs = 4
        tau = 0.1
        sigma_min, sigma_max = 0.01, 100
        sigmas = K.sampling.get_sigmas_karras(25, sigma_min, sigma_max, device=device)
        input_ids = input_ids[:bs, :n_tokens]
        attention_mask = attention_mask[:bs, :n_tokens]
        x = torch.randn([bs, args.z_dim], device=device) * sigma_max
        z_prev = model_vae.encode(input_ids, attention_mask)[:, None]
        extra_args = {
            "z_prev": z_prev,
            "padding_mask": torch.ones([bs, 1], dtype=torch.long, device=device),
        }
        mean = K.sampling.sample_dpmpp_2m_sde(
            model, x, sigmas, eta=0.0, extra_args=extra_args, disable=not is_main
        )
        z = model_vae.vae.sample(mean, tau=tau)
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
            n = input_ids.shape[0]

            if i % demo_every == 0:
                demo(model_ema, input_ids, attention_mask, args.context)

            with accelerator.accumulate(model), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                input_ids = input_ids[:, : ctx_len * args.context]
                attention_mask = attention_mask[:, : ctx_len * args.context]

                input_ids_in = rearrange(input_ids, "n (chunk s) -> (n chunk) s", s=args.context)
                attention_mask_in = rearrange(attention_mask, "n (chunk s) -> (n chunk) s", s=args.context)
                z_all = model_vae.encode(input_ids_in, attention_mask_in)
                z_all = rearrange(z_all, "(n chunk) d -> n chunk d", n=n)
                z_prev, z = z_all[:, :-1], z_all[:, -1]

                if measure_ae_scale:
                    # TODO: measure AE scale in the AE training code instead
                    ae_mean_sq = z_prev.detach().pow(2).mean()
                    ae_mean_sq = accelerator.reduce(ae_mean_sq, "mean")
                    ae_scale_sum += ae_mean_sq
                    ae_scale_avg = torch.sqrt(ae_scale_sum / (i + 1))
                    print0("ae_scale_avg", ae_scale_avg.item(), file=sys.stderr)
                    if i == 100:
                        return

                noise = torch.randn_like(z)
                sigma = K.utils.rand_v_diffusion(
                    (n,), ae_scale, min_value=1e-3, max_value=1e3, device=device
                )
                # padding_mask = torch.ones([n, ctx_len - 1], dtype=torch.long, device=device)
                u = torch.randint(ctx_len, [n], device=device)
                padding_mask = 1 - F.one_hot(u, ctx_len).flip(1).cumsum(dim=1).flip(1)[:, 1:]
                losses = model.loss(z, noise, sigma, z_prev=z_prev, padding_mask=padding_mask)
                loss = torch.mean(losses)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                gns = 0.0
                if gns_stats is not None:
                    norm_small, norm_large = gns_stats_hook.get_stats()
                    gns_stats.update(norm_small, norm_large, n, n * accelerator.num_processes)
                    gns = gns_stats.get_gns()

                opt.step()
                opt.zero_grad()
                if accelerator.sync_gradients:
                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update(model, model_ema, ema_decay)
                ema_sched.step()

                loss_global = accelerator.reduce(loss, "mean").item()
                ema_update_dict(ema_stats, {"loss": loss_global}, ema_decay)
                loss_avg = ema_stats["loss"]
                print0(f"epoch: {epoch}, step: {i}, loss: {loss_global:g}, avg loss: {loss_avg:g}, gns: {gns:g}", file=sys.stderr)
                i += 1

                if i % args.save_every == 0:
                    save()

        save()


if __name__ == "__main__":
    main()
