from typing import Literal

import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from xent.config import * 

class M():

    """ manage model loading and tokenization """

    def __init__(
            self, 
            model_name: str, 
            model_version: str,
            base: str = "base",
            reinitialize: bool = False
            ):
        
        self.model_name = model_name
        self.model_version = model_version
        self.base = base

        task_models_dir = os.path.join(work_dir, "models", base)
        model_path = os.path.join(task_models_dir, model_name, model_version)     

        base_models_dir = os.path.join(work_dir, "models", "base")
        tokenizer_path = os.path.join(base_models_dir, model_name, "M0") # tokenizer is contained in the original version
        
        if reinitialize: # load only model shape and randomly initialized weights
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_config(config).to(device)
        else: # load the from-pretrained model 
            if model_version == "M0":
                self.model = self.load_origin_model(model_path)
            else: 
                model_path = os.path.join(model_path, model_version)
                self.model = self.load_torch_model(model_path)
        
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        config_path = os.path.join(tokenizer_path, "config.json")
        self.config = self.load_model_config(config_path)
        self.vocab_size = self.tokenizer.vocab_size
        if model_name.startswith("gpt"):
            self.ctx_window = self.config["n_ctx"]
        else:
            print("Model initialized without context_window.")
            print("Use the set_context_window method to set it or you'll get errors somewhere.")
        
        

    def tokenize(
            self, 
            text: str,
            padding="do_not_pad"
            ) -> dict[torch.Tensor, torch.Tensor]:
        """ The tokenize method returns a dictionary with following keys: 
            input_ids:      Tensor with token keys
            attention_mask: Tensor with the attention masking """
        return self.tokenizer(
            text, 
            return_tensors="pt",
            padding=padding,
            max_length=self.ctx_window
        ).to(device)
    
    def detokenize(self, tokens, mode: Literal["single", "tensor", "list", "full_batch"]="tensor") -> str:
        """ single and list mode work only on a single sample. batch is for multiple batches """
        modes = ["single", "tensor", "list", "full_batch"]
        if mode not in modes:
            raise ValueError(f"mode {mode} is not in available modalities: {', '.join(modes)}")
        if mode == "single":
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        elif mode == "tensor":
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        elif mode == "full_batch":
            #TODO check the behavior of batch mode
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        elif mode == "list":
            return [self.tokenizer.decode([tok], skip_special_tokens=True) for tok in tokens[0]]    
    
    def get_xent(
            self, 
            input, 
            starting_index=None,
            reduction="none"
            ):
        if isinstance(input, str): tokens = self.tokenize(input).input_ids
        else: tokens = input
        logits = self.model(tokens).logits
        if starting_index:
            return F.cross_entropy(logits[0, starting_index:-1], tokens[0, starting_index+1:], reduction=reduction)
        else: return F.cross_entropy(logits[0, :-1], tokens[0, 1:], reduction=reduction)

    def set_context_window(self, value:int):
        self.ctx_window = value

    @staticmethod
    def load_origin_model(path:str):
        return AutoModelForCausalLM.from_pretrained(path).to(device) 

    @staticmethod
    def load_torch_model(path:str):
        return torch.load(path, weights_only=False, map_location=device)

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