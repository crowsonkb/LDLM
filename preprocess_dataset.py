from argparse import ArgumentParser
import random
import json
import zipfile
from pathlib import Path
from concurrent import futures
from itertools import chain, islice
import os
import sys

from transformers import AutoTokenizer
import safetensors.torch as safetorch
import torch
from torch.utils import data
from datasets import load_dataset
from tqdm import trange, tqdm


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
        

def batch_to_tensors(batch, tokenizer, context, device="cpu"):
    batch = [i for i in batch if len(i) < 250000]
    def tokwrap(item):
        """Wrap the tokenizer so we can try-catch."""
        try:
            return tokenizer(item)
        except:
            return None
    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        document_tokens = [inputs.input_ids for inputs in executor.map(tokwrap, batch)
                           if inputs]
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
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-dataset",
                        default="togethercomputer/RedPajama-Data-1T-Sample",
                        help="bulk pretraining dataset to tune on")
    parser.add_argument(
        "--model",
        type=str,
        default="openlm-research/open_llama_3b_v2",
        help="model name",
    )
    parser.add_argument("--context", type=int, default=2048, help="context window length")
    parser.add_argument("--output", default="RedPajama-Data-1T-Sample-Preprocessed")
    args = parser.parse_args()

    if Path(args.model).exists():
        model_name = Path(args.model).resolve()
    else:
        model_name = args.model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pretraining_dataset = load_dataset(args.pretraining_dataset)
        
    pretraining_dataloader = data.DataLoader(
        pretraining_dataset['train']['text'],
        batch_size=32,
    )

    pretraining_preprocessed = []
    for batch in tqdm(pretraining_dataloader):
        pretraining_preprocessed.append(batch_to_tensors(batch,
                                                         tokenizer,
                                                         args.context))
        pretraining_tokens = sum([i[0].shape[0] * i[0].shape[1]
                                  for i in pretraining_preprocessed])

    pt_inputs = torch.cat([i[0] for i in pretraining_preprocessed])
    pt_masks = torch.cat([i[1] for i in pretraining_preprocessed])

    os.mkdir(args.output)
    rows_saved = 0
    shard_index = 0
    while rows_saved < pt_inputs.shape[0]:
        tensors = {
            "input_ids": pt_inputs[rows_saved:rows_saved+64000].int(),
            "attention_mask": pt_masks[rows_saved:rows_saved+64000].byte(),
        }
        safetorch.save_file(tensors,
                            os.path.join(args.output,
                                         f'shard_{shard_index}.safetensors')
        )
        rows_saved += 64000
        shard_index += 1

if __name__ == "__main__":
    main()
