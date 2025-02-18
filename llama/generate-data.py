import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from unsloth import FastLanguageModel
from xent import X
from xent.config import *

# definition of the device
device = "cuda:3"

# initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
)

# load wikipedia to retrieve corpus pieces and shard it into a smaller chunk
data = load_dataset(
    "wikipedia", 
    "20220301.en", 
    trust_remote_code=True,
    )["train"].shard(100, 51)

# here's a simple tokenizer wrapper to not repeat arguments all around
def tokenize(text, add_bos=False, **kwargs):
        return tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_bos
        )["input_ids"]

def save_checkpoint(dataset, dataset_dir, checkpoint_num):
    checkpoint_file = os.path.join(dataset_dir, f"checkpoint_{checkpoint_num}.json")
    print(f"Saving checkpoint number {checkpoint_num} into {checkpoint_file}")
    with open(checkpoint_file, "w") as f:
        json.dump(dataset, f)
    return []

# make the directory where we'll be saving data checkpoints
output_dir = os.path.join(data_dir, "instruct_llama")
closure_dir = os.path.join(output_dir, "closure-test")
ranking_dir = os.path.join(output_dir, "ranking-test")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(closure_dir, exist_ok=True)
os.makedirs(ranking_dir, exist_ok=True)

# define the max window we want the corpus to be, since wiki articles are pretty long
corpus_window = 256

# tokenize structural entities for generating tasks
colon = tokenize(":").to(device)
nline = tokenize("\n").to(device)
major = tokenize(">").to(device)
xent_closure_toks = tokenize(
        X.xmap("\ndef fwd_closure(integer):\n")
    ).to(device)
xent_ranking_toks = tokenize(
        X.xmap("\ndef xent_rank():\n")
    ).to(device)

# get the xent places to divide prompt and response
x1 = X.xdef
x2 = X.xreturn
ln1 = len(x1)
ln2 = len(x2)

# initialize empty list to add dicts of prompt response
closure_dataset = []
ranking_dataset = []

# make sure there's torch nograd or the GPU will explode
with torch.no_grad():
    
    i = 0
    j = 0
    
    for d in tqdm(data):
        
        tokens = tokenize(d["text"]).to(device)
        window = min(corpus_window, tokens.shape[1])
        tokens = tokens.unfold(1, window, window).squeeze(0)
        
        for slice in tokens:
            
            slice = slice.unsqueeze(0)
            logits = model(slice).logits
            xent = F.cross_entropy(logits[:, :-1].view(-1, logits.shape[-1]), slice[:, 1:].view(-1), reduction="none")
            
            new_sample = torch.cat([slice, xent_closure_toks], dim=-1)
            for tok, xnt in zip(slice[0, 1:], xent):
                xentok_i = tokenize(f" {round(float(xnt))}").to(device)
                new_sample = torch.cat([new_sample, tok.view(1,1), colon, xentok_i, nline], dim=-1)
            text = tokenizer.decode(new_sample.squeeze(0))
            idx2 = text.find(x2) + ln2
            closure_dataset.append(
                {
                    "prompt": text[:idx2],
                    "response": text[idx2:]
                }
            )

            new_sample = torch.cat([slice, xent_ranking_toks], dim=-1)
            ranked_xents, sorting = torch.sort(xent, descending=True)
            sorted_toks = slice[0, 1:][sorting]
            task_toks = torch.cat([
                sorted_toks.view(sorted_toks.shape[0], 1),
                # torch.cat([tokenize(f" {round(float(x))}") for x in ranked_xents]).to(device), # uncomment if you also want to put xent in the output task
                major.squeeze(0).repeat(sorted_toks.shape[0], 1)
            ], dim = 1)
            new_sample = torch.cat([new_sample, task_toks.view(1,-1)], dim=-1)
            text = tokenizer.decode(new_sample.squeeze(0))
            idx2 = text.find(x2) + ln2
            ranking_dataset.append(
                {
                    "prompt": text[:idx2],
                    "response": text[idx2:]
                }
            )

            i += 1
            
            if i % 1000 == 0:
                j += 1
                closure_dataset = save_checkpoint(closure_dataset, closure_dir, j)
                ranking_dataset = save_checkpoint(ranking_dataset, ranking_dir, j)

