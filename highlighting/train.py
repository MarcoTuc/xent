import sys
import os 
home_dir = os.path.expanduser("~")
work_dir = os.path.join(home_dir, "synth", "highlighting")
import logging
logging.getLogger().setLevel(logging.INFO)
import warnings
warnings.filterwarnings("ignore")

import time
import pickle
import json
from tqdm import tqdm

# ML AND SCI LIBRARIES
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss

# XENT code
from xentlang import X
from utils import Tee

device = torch.device("cuda:3")
models_path = os.path.join(work_dir, "models")
data_path = os.path.join(work_dir, "data")

# define initial model version and new version
model_name = "gpt2"
model_version = "M0"
new_model_version = "M1"

# utility parameters
cut_dataset = None
log_in_text = True

# Hyperparameters
LEARNING_RATE = 6e-4 # take it from Karpathy nano-GPT 
EPOCHS = 15
# TODO add all the available hyperparameters
data_split = 0.6 # train/test ratio
batch_size = 10

beta1 = 0.1
beta2 = 0.95
grad_clip = 1.0


def load_model_and_tokenizer(path: str):
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    return model

def load_torch_model(path: str):
    model = torch.load(path, weights_only=False)
    return model 

def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# DATA LOADING METHOD
def load_dataset(name: str):
    with open(os.path.join(data_path, f"{name}.pkl"), "rb") as data:
        return pickle.load(data)
    
class TextDataset(Dataset):
    def __init__(self, dataset: list[str], tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = [self.tokenize(text) for text in tqdm(dataset)]

    def tokenize(self, text): 
        return self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(device)
    
    def tokenize_single(self, text):
        return self.tokenizer(
            text, 
            return_tensors="pt",
            padding=True
        ).to(device)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> str:
        return self.dataset[index]

def find_xent_def(tokens, return_len=False):
    """ Returns the index at which the xent function starts, needed for starting the loss computation """
    xdefseq = tokenizer.encode(X.xreturn, return_tensors="pt").to(device)
    seq_len = xdefseq.shape[1]
    windows = tokens.input_ids.unfold(dimension=2, size=seq_len, step=1)
    matches = (windows==xdefseq).all(dim=3)
    indices = matches.nonzero().squeeze(0)
    if return_len:
        return indices, seq_len
    return indices

# load the model
path = os.path.join(models_path, model_name, model_version)
M0, tokenizer = load_model_and_tokenizer(path)

# load the data
D0 = load_dataset("D0")
if cut_dataset:
    D0 = D0[:cut_dataset]

train_size = int(data_split * len(D0))
test_size = len(D0) - train_size

train_dataset, test_dataset = random_split(D0, [train_size, test_size])
print("Tokenizing training set:")
train_dataset = TextDataset(train_dataset, tokenizer)
print("Tokenizing test set:")
test_dataset = TextDataset(test_dataset, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
gen_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

crossentropy = CrossEntropyLoss()
optimizer = AdamW(M0.parameters(), lr = LEARNING_RATE, betas=(beta1, beta2), eps=1e-9, weight_decay=0.01)
lr_lambda = lambda epoch: 0.965 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)

loss_series = []
val_series = []
log_interval = 10

def train(model):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch, tokens in enumerate(train_loader):
        optimizer.zero_grad()
        try: xidx, xlen = find_xent_def(tokens, return_len=True)
        except Exception as e: 
            print(f"Incurred in find_xent_def error: {e}")
            print("This is because the piece of data at hand has incurred in an old bug that depends on the generation procedure.")
            print("Skipping this datapoint and moving on to the next!")
            continue
        tokens, attn_mask = tokens.input_ids, tokens.attention_mask # [B, T]
        logits = model(input_ids=tokens, attention_mask=attn_mask).logits  # [B, T, L]
        loss = 0
        for sample, _, fstart in xidx:
            xstart = fstart + xlen
            sample_logits = logits[sample, :, xstart:-1].view(-1, logits.size(-1)) # [T, L]
            sample_targets = tokens[sample, :, xstart+1:].view(-1) # [T]
            sample_loss = crossentropy(sample_logits, sample_targets)
            loss += sample_loss
        loss = loss / logits.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.detach().data
        if (batch + 1) % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            loss_series.append(float(cur_loss))
            elapsed = time.time() - start_time
            print(f"| batch: {batch+1} | loss: {cur_loss:.5f} | has taken: {elapsed:.2f} seconds")
            generate_in_loop(model)
            total_loss = 0
            start_time = time.time()

def evaluate(test_model, test_loader):
    test_model.eval()
    total_val_loss = 0
    nbatches = min(100, test_size)
    print(f"number of evaluation batches: {nbatches}")
    with torch.no_grad():
        for batch, tokens in enumerate(test_loader):
            try: xidx, xlen = find_xent_def(tokens, return_len=True)
            except Exception as e: 
                print(f"Incurred in find_xent_def error: {e}")
                print("This is because the piece of data at hand has incurred in an old bug that depends on the generation procedure.")
                print("Skipping this datapoint and moving on to the next!")
                continue
            tokens, attn_mask = tokens.input_ids, tokens.attention_mask
            logits = test_model(input_ids=tokens, attention_mask=attn_mask).logits
            loss = 0
            for sample, _, fstart in xidx:
                xstart = fstart + xlen
                sample_logits = logits[sample, :, xstart:-1].view(-1, logits.size(-1))
                sample_targets = tokens[sample, :, xstart+1:].view(-1)
                sample_loss = crossentropy(sample_logits, sample_targets)
                loss += sample_loss
            loss = loss / logits.shape[0]
            total_val_loss += loss 
            if batch > nbatches:
                break
    return total_val_loss / nbatches

def generate_in_loop(gen_model):
    gen_model.eval()
    tokens = next(iter(gen_loader))
    xidx, xlen = find_xent_def(tokens, return_len=True)
    try: 
        xstart = xidx[2] + xlen
        prompt = tokens.input_ids[0, :, :xstart]
        attn_mask = tokens.attention_mask[0, :, :xstart]
    except Exception as e: 
        print(f"xidx fuckup  {e}")
        print(f"xidx tensor: {xidx}")
        print(f"xidx shape:  {xidx.shape}")
        return
    
    with torch.no_grad():
        gen = gen_model.generate(
            prompt, 
            attention_mask=attn_mask,
            do_sample=True, 
            temperature=1.0, 
            pad_token_id = gen_model.config.eos_token_id,
            max_length = 1024,
            )
        print("\n------------------------------ GENERATION SAMPLE ------------------------------\n")
        print(f"--------- prompt --------- | shape: {prompt.shape}")
        dprompt = tokenizer.decode(gen[0, :prompt.shape[1]], skip_special_tokens=True)
        print(dprompt)
        print("--------- output ---------\n")
        gen = tokenizer.decode(gen[0, prompt.shape[1]:], skip_special_tokens=True)
        print(gen)
        print("\n-----------------------------------  NEXT  ------------------------------------\n")

    gen_model.train()

best_loss = float("inf")
best_model = None

if new_model_version == model_version:
    raise NameError(f"New model version {new_model_version} should be different than the old model version {model_version}")
model_save_folder = os.path.join(models_path, model_name, new_model_version)
model_save_path = os.path.join(model_save_folder, new_model_version)
os.makedirs(model_save_folder, exist_ok=True)

if log_in_text:
    f = open(os.path.join(model_save_folder, "console.txt"), "w+")
    sys.stdout = Tee(f)

for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch}/{EPOCHS}")
    train(M0)
    print("Evaluating...", end=" ")
    val_loss = evaluate(M0, test_loader=test_loader)
    val_series.append(float(val_loss))
    print(f"Validation loss: {val_loss:.5f}")

    if val_loss < best_loss:
        best_loss = val_loss
        print("Saving new best model...", end=" ")
        torch.save(best_model, model_save_path)
        print("Model saved!", end=" ")
    
    scheduler.step()

with open(os.path.join(model_save_folder, "training_details.json"), "w") as js:
    json.dump(
        {
            "test_log_interval": log_interval,
            "train_size_aka_val_interval": train_size,
            "loss_series": loss_series,
            "val_series": val_series
        },
        js
    )
print("Details saved!")

# wandb to plot things during training 