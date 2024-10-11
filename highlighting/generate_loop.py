import os 
from generate import generate_dataset, save_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = torch.device("cuda:3")
def load_model_and_tokenizer(path: str):
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model_name = "gpt2"
model_version = "M0"
model_path = os.path.join("highlighting", "models", model_name, model_version)
model, tokenizer = load_model_and_tokenizer(model_path)

dataset_size = 50000
new_db = generate_dataset(model, tokenizer, dataset_size=dataset_size)

dataset_name = "D0-big.pkl"
save_path = os.path.join("highlighting", "data", dataset_name)
save_dataset(save_path, new_db)