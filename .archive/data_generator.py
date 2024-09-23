import pickle
import random

import torch 
import torch.nn.functional as F

from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from functions import *

device = torch.device("cuda")
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(text): return tokenizer(text, return_tensors="pt", padding=True).to(device)
def detokenize(tokens): return tokenizer.decode(tokens[0], skip_special_tokens=True)

def generate_sample(
        max_new_tokens=660,
        temperature=1.2
        ):
    input = tokenize("\n")
    output = model.generate(
        **input,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    return detokenize(output)

def pickYX(sample):
    tokens = tokenize(sample)["input_ids"][0]
    

print(pickYX(generate_sample()))

