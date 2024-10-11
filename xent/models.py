import os
from typing import Literal
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import * 

class M:

    """ manage model loading and tokenization """

    models_dir = os.path.join(work_dir, "models")

    def __init__(
            self, 
            model_name: str, 
            model_version: str
            ):

        model_path = os.path.join(self.models_dir, model_name, model_version)
        tokenizer_path = os.path.join(self.models_dir, model_name, "M0") # tokenizer is contained in the original version
        
        if model_version == "M0":
            self.model = self.load_origin_model(model_path)
        else: 
            model_path = os.path.join(model_path, model_version)
            self.model = self.load_torch_model(model_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        
        config_path = os.path.join(model_path, "config.json")
        self.config = self.load_model_config(config_path)
        self.vocab_size = self.tokenizer.vocab_size
        if model_name.startswith("gpt"):
            self.context_window = self.config["n_ctx"]
        else:
            print("Model initialized without context_window.")
            print("Use the set_context_window method to set it.")

    def tokenize(self, text: str) -> dict[torch.Tensor, torch.Tensor]:
        """ The tokenize method returns a dictionary with following keys: 
            input_ids:      Tensor with token keys
            attention_mask: Tensor with the attention masking """
        return self.tokenizer(
            text, 
            return_tensors="pt",
            # padding="max_length",
            # truncation=True,
            # max_length=self.context_window
        ).to(device)
    
    def detokenize(self, tokens, mode: Literal["single", "batch", "list"]) -> str:
        modes = ["single", "batch", "list"]
        if mode not in modes:
            raise ValueError(f"mode {mode} is not in available modalities: {','.join(modes)}")
        if mode == "single":
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        elif mode == "batch":
            #TODO check the behavior of batch mode
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        elif mode == "list":
            return [self.tokenizer.decode(tok, skip_special_tokens=True) for tok in tokens]    
    
    def set_context_window(self, value:int):
        self.context_window = value

    @staticmethod
    def load_origin_model(path:str):
        return AutoModelForCausalLM.from_pretrained(path).to(device) 

    @staticmethod
    def load_torch_model(path:str):
        return torch.load(path, weights_only=False).to(device)

    @staticmethod
    def load_tokenizer(path:str):
        tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @staticmethod
    def load_model_config(path:str):
        with open(path, "r+") as f:
            return json.load(f)

    @staticmethod
    def save_model(path:str):
        """ #TODO add a modelsaving method in the model class """
        pass