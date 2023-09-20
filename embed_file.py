#!/usr/bin/env python3

"""Trains a latent diffusion language model."""

import argparse
from itertools import chain, islice
import json
import math
from pathlib import Path
import random
import sys
import zipfile

import accelerate
import safetensors.torch as safetorch
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from vae_model import DecoderOnlyTransformerVAE

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
    parser.add_argument("--input-text", type=Path, required=True, help="input text file")
    # parser.add_argument("--batch-size", type=int, default=4, help="microbatch size")
    parser.add_argument(
        "--model",
        type=str,
        default="openlm-research/open_llama_3b_v2",
        help="model name",
    )
    parser.add_argument("--context", type=int, default=48, help="context window length")
    parser.add_argument("--output", type=Path, required=True, help="path to save embeds")
    parser.add_argument("--rank", type=int, default=32, help="the lora rank")
    parser.add_argument("--z-dim", type=int, default=768, help="the latent dimension")
    parser.add_argument("--vae", type=Path, required=True, help="the vae checkpoint to use")
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(mixed_precision="bf16")
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
            model_name, device, z_dim=args.z_dim, lora_rank=args.rank
        )

    model_vae.load_pretrained(args.vae)

    i = 0
    zs = []
    n = 1

    text = args.input_text.read_text()
    input_ids, attn_mask = batch_to_tensors([text], tokenizer, args.context, device=device)

    with tqdm(total=n, disable=not is_main) as pbar:
        input_ids = input_ids.long()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            z = model_vae.encode(input_ids, attn_mask)
            zs.append(z)
            i += zs[-1].shape[0]
            pbar.update(zs[-1].shape[0])

    zs = torch.cat(zs, dim=0)
    safetorch.save_file({"embeds": zs}, args.output)


if __name__ == "__main__":
    main()
