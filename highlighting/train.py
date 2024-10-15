# Main training loop here, comments down below. 

# OS IMPORTS 
import sys
import os 
import logging
logging.getLogger().setLevel(logging.INFO)
import warnings
warnings.filterwarnings("ignore")

# GENERAL PYTHON IMPORTS
import time
import pickle
import json
from tqdm import tqdm

# ML AND SCI LIBRARIES
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss

# CUSTOM IMPORTS
from xentlang import X
from utils import Tee

# Set the device and standart paths for models and data
device = torch.device("cuda:3")
work_dir = os.path.join(os.getcwd(), "highlighting")
models_path = os.path.join(work_dir, "models")
data_path = os.path.join(work_dir, "data")

# define initial model version and new version
model_name = "gpt2"
model_version = "M0"
new_model_version = "M1-test"
# just check you are not overwriting an existing model
if new_model_version == model_version:
    raise NameError(f"New model version {new_model_version} should be different than the old model version {model_version}")
# make directories for the new model
model_save_folder = os.path.join(models_path, model_name, new_model_version)
model_save_path = os.path.join(model_save_folder, new_model_version)
try: os.makedirs(model_save_folder, exist_ok=False)
except OSError: raise OSError(f"Please provide a different name to the new model. {new_model_version} already exists for {model_name}")

# call the synthetic dataset you use for training
synthetic_dataset_name = "D0-big"

# utility parameters
cut_dataset = 5000 # if you want to cut the dataset for some reason.
log_in_text = True # log training in a txt file. It will be inside models/{model_name}/{new_model_version}

# training loop size
EPOCHS = 15 # I did 15 but it looks like too many, I would put it down to 5-6 as valid loss usually starts growing at epoch 4-5
data_split = 0.6 # train/test ratio
batch_size = 10 # how many batches per training loop
log_interval = 10 # how many iterations until you report something. you get a report after log_interval*batch_size datapoints trained

# optimization parameters
LEARNING_RATE = 6e-4 # took it from Karpathy nanoGPT 
beta1 = 0.1
beta2 = 0.95
grad_clip = 1.0

# MODEL LOADING METHODS
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

# DATASET LOADING METHOD
def load_dataset(name: str):
    with open(os.path.join(data_path, f"{name}.pkl"), "rb") as data:
        return pickle.load(data)

# DATAPROCESSING CLASS
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

# We train only on the output after prompt+xent_function. This helps to find the index at which you start computing the loss. 
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

##############################################################################################################################

# load the model
path = os.path.join(models_path, model_name, model_version)
M0, tokenizer = load_model_and_tokenizer(path)

# load the data
D0 = load_dataset(synthetic_dataset_name)
if cut_dataset:
    D0 = D0[:cut_dataset]

# train split the data and make dataloaders 
train_size = int(data_split * len(D0))
test_size = len(D0) - train_size
train_dataset, test_dataset = random_split(D0, [train_size, test_size])
print("Tokenizing training set:")
train_dataset = TextDataset(train_dataset, tokenizer)
print("Tokenizing test set:")
test_dataset = TextDataset(test_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
gen_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) # used to generate text samples you can see while training

# initialize optimizator
crossentropy = CrossEntropyLoss()
optimizer = AdamW(M0.parameters(), lr = LEARNING_RATE, betas=(beta1, beta2), eps=1e-9, weight_decay=0.01)
lr_lambda = lambda epoch: 0.965 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)

# record what's going on
loss_series = []
val_series = []

# training loop 
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
        # loop down here is to train only on the output of the xent function
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

# evaluation loop
def evaluate(test_model, test_loader):
    test_model.eval()
    total_val_loss = 0
    nbatches = min(70, test_size)
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

# generate while training
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
            "train_size": train_size,
            "batch_size": batch_size,
            "loss_series": loss_series,
            "val_series": val_series
        },
        js
    )
print("Details saved!")