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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss

# XENT code
from highlight import XentLang as X
from utils import Tee

device = torch.device("cuda:3")
models_path = os.path.join(work_dir, "models")
data_path = os.path.join(work_dir, "data")

# utility parameters
cut_dataset = None

# Hyperparameters
LEARNING_RATE = 6e-4 # take it from Karpathy nano-GPT 
EPOCHS = 1
# TODO add all the available hyperparameters
data_split = 0.6 # train/test ratio

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

def find_xent_def(tokens):
    """ Returns the index at which the xent function starts, needed for starting the loss computation """
    xdefseq = tokenizer.encode(X.xdef, return_tensors="pt").to(device)
    seq_len = xdefseq.shape[1]
    windows = tokens.input_ids.unfold(dimension=2, size=seq_len, step=1)
    matches = (windows==xdefseq).all(dim=3)
    indices = matches.nonzero().squeeze(0)
    return indices
    

# load the model
path = os.path.join(models_path, "gpt2-xl-M0")
M0, tokenizer = load_model_and_tokenizer(path)

# load the data
D0 = load_dataset("D0")
if cut_dataset:
    D0 = D0[:cut_dataset]

train_size = int(data_split * len(D0))
test_size = len(D0) - train_size

train_dataset, test_dataset = D0[:train_size], D0[train_size:]
print("Tokenizing training set:")
# train_dataset = train_dataset[1166]
train_dataset = TextDataset(train_dataset, tokenizer)
# print("Tokenizing test set:")
# test_dataset = TextDataset(test_dataset, tokenizer)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

crossentropy = CrossEntropyLoss()
optimizer = AdamW(M0.parameters(), lr = LEARNING_RATE, betas=(beta1, beta2), eps=1e-9, weight_decay=0.01)
lr_lambda = lambda epoch: 0.965 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)

# TODO: make it such that loss is computed only after the xent function is called. 

loss_series = []
val_series = []
log_interval = 10
# prev_tokens = None

def train(model):
    # model.train()
    # total_loss = 0
    # start_time = time.time()
    for batch, tokens in enumerate(train_loader):
        # optimizer.zero_grad()
        try: 
            xidx = find_xent_def(tokens)[2]
            # print(f"xidx at batch {batch} successful")
            # print(f"Tokens shape: {tokens.shape}\n--------------------------")
            # prev_tokens = tokens
        except Exception as e:
            # print(f"batch {batch} failed") 
            print(f"Incurred in find_xent_def error at batch {batch}: {e}")
            print(find_xent_def(tokens))
            print(tokens)
            # print(find_xent_def(prev_tokens))
            # print(prev_tokens)
            print(tokens.input_ids.shape)
            print(tokenizer.decode(tokens.input_ids[0][0]))
            print("\n-------------------------------------------\n")
            # print(find_xent_def(prev_tokens))
            # print("Couldn't retrieve xidx: this is the returned value of the function:")
            # print(xidx)
            # print("The tokens tensor had this shape")
            # print(tokens.input_ids.shape)
            # print("And these are the tokens we fed it:")
            # print(tokens.input_ids)
            # print("---------------------------------------------")
            continue
        # tokens, attn_mask = tokens.input_ids.view(1, -1), tokens.attention_mask.view(1, -1) # [B, T]
        # logits = model(input_ids=tokens, attention_mask=attn_mask).logits  # [B, T, L]
        # loss = crossentropy(logits[:, xidx:][:-1], tokens[:, xidx:][1:]) # old loss but it's still doing the right thing
        # loss = crossentropy(logits[:, xidx:-1].view(-1, logits.shape[-1]), tokens[:, xidx+1:].view(-1))
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # optimizer.step()
        # total_loss += loss.item() 
        # if (batch + 1) % log_interval == 0 and batch > 0:
            # cur_loss = total_loss / log_interval
            # loss_series.append(float(cur_loss))
            # elapsed = time.time() - start_time
            # print(f"| batch: {batch+1} | loss: {cur_loss:.5f} | has taken: {elapsed:.2f} seconds")
            # total_loss = 0
            # start_time = time.time()


# def evaluate(test_model, test_loader):
#     test_model.eval()
#     total_val_loss = 0
#     nbatches = min(100, test_size)
#     print(f"number of evaluation batches: {nbatches}")
#     with torch.no_grad():
#         for batch, tokens in enumerate(test_loader):
#             try: xidx = find_xent_def(tokens)[2]
#             except Exception as e: 
#                 print(f"Incurred in find_xent_def error: {e}")
#                 print("Couldn't retrieve xidx: this is the returned value of the function:")
#                 print(xidx)
#                 print("The tokens tensor had this shape")
#                 print(tokens.input_ids.shape)
#                 print("And these are the tokens we fed it:")
#                 print(tokens.input_ids)
#                 continue
#             tokens, attn_mask = tokens.input_ids.view(1, -1), tokens.attention_mask.view(1, -1)
#             logits = test_model(input_ids=tokens, attention_mask=attn_mask).logits
#             loss = crossentropy(logits[0, xidx:-1].view(-1, logits.shape[-1]), tokens[0, xidx+1:].view(-1))
#             total_val_loss += loss 
#             # print(f"adding_loss: {loss:.3f}")
#             if batch > nbatches:
#                 break
#     return total_val_loss / nbatches

best_loss = float("inf")
best_model = None


new_model_name = "gpt2-xl-M1"
model_save_folder = os.path.join(models_path, new_model_name)
model_save_path = os.path.join(model_save_folder, new_model_name)
os.makedirs(model_save_folder, exist_ok=True)

# f = open(os.path.join(model_save_folder, "debug.txt"), "w+")
# sys.stdout = Tee(f)

for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch}/{EPOCHS}")
    train(M0)
    # print("Evaluating...", end=" ")
    # val_loss = evaluate(M0, test_loader=test_loader)
    # val_series.append(float(val_loss))
    # print(f"Validation loss: {val_loss:.5f}")

    # if val_loss < best_loss:
        # best_loss = val_loss
        # best_model = M0
    
    # scheduler.step()

# print("Saving the model...", end=" ")
# torch.save(M0, model_save_path)
# print("Model saved!", end=" ")

# with open(os.path.join(model_save_folder, "training_details.json"), "w") as js:
    # json.dump(
        # {
            # "test_log_interval": log_interval,
            # "test_size_aka_val_interval": test_size,
            # "loss_series": loss_series,
            # "val_series": val_series
        # },
        # js
    # )
# print("Details saved!")

# wandb to plot things during training 