import os 
work_dir = os.path.join(os.getcwd(), "highlighting")
import logging
logging.getLogger().setLevel(logging.INFO)
import warnings
warnings.filterwarnings("ignore")

import pickle
from tqdm import tqdm

# ML AND SCI LIBRARIES
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from generate import generate_dataset, save_dataset

device = torch.device("cuda:0")
models_path = os.path.join(work_dir, "models")
data_path = os.path.join(work_dir, "data")

print(models_path)

# LOAD THE MODEL 
def load_model_and_tokenizer(path: str):
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=True)
    return model, tokenizer

def load_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    return model

path = "models/gpt2-xl-M0"
M0, tokenizer = load_model_and_tokenizer(path)

# GENERATE THE DATASET
era = 0
D0 = generate_dataset(2000)
save_dataset(f"{data_path}/D{era}.pkl", D0)

# LOAD THE DATASET
class TextDataset(Dataset):
    def __init__(self, dataset: list[str], tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> str:
        return self.dataset[index]


data_split = 0.7
train_size = int(data_split * len(D0))
test_size = len(D0) - train_size

train_dataset, test_dataset = random_split(D0, [train_size, test_size])
train_dataset = TextDataset(train_dataset, tokenizer)
test_dataset = TextDataset(test_dataset, tokenizer)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# TRAIN THE MODEL

# EVALUATE THE MODEL