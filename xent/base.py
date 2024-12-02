import os
import json
import random
import pickle

import torch
import torch.nn.functional as F

from typing import Literal, Callable
from tqdm import tqdm

from torch.utils.data import Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from xent.config import *

#######################################################################################################
################################## MODELS #############################################################
#######################################################################################################

class M():

    """ manage model loading and tokenization """

    def __init__(
            self, 
            base=None,
            model_name=None, 
            model_version=None,
            reinitialize=False,
            init_from_path=None
            ):
        
        if base and not init_from_path: self.base = "base"

        self.model_name = model_name
        self.model_version = model_version
        self.base = base

        task_models_dir = os.path.join(models_dir, base)
        model_path = os.path.join(task_models_dir, model_name, model_version)     

        base_models_dir = os.path.join(models_dir, "base")
        tokenizer_path = os.path.join(base_models_dir, model_name, "M0") # tokenizer is contained in the original version
        
        print("tokenizer path:", tokenizer_path)
        print("model path:", model_path)
        
        config = AutoConfig.from_pretrained(tokenizer_path)
        config.use_cache = False # disable KV-caching during training

        if reinitialize: # load only model shape and randomly initialized weights  
            self.model = AutoModelForCausalLM.from_config(config).to(device)
        else: # load the from-pretrained model 
            if model_version == "M0":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    config=config
                ).to(device)
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

        self.model.gradient_checkpointing_enable()

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
            
    def pad(self, tensor):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        pad_length = self.ctx_window - tensor.shape[1]
        if pad_length > 0:
            padding = torch.full((tensor.shape[0], pad_length), self.tokenizer.pad_token_id).to(device)
            return torch.cat([tensor, padding], dim=1)
        return tensor

    def get_xent(
            self, 
            input:torch.Tensor, 
            starting_index=0,
            reduction="none"
            ) -> torch.Tensor:
        if isinstance(input, str): tokens = self.tokenize(input).input_ids
        else: tokens = input
        logits = self.model(tokens).logits
        return F.cross_entropy(logits[0, starting_index:-1], tokens[0, starting_index+1:], reduction=reduction)

    def set_context_window(self, value:int):
        self.ctx_window = value

    @staticmethod
    def load_origin_model(path:str):
        return AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16).to(device) 

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

#######################################################################################################
################################## TASKS ##############################################################
#######################################################################################################

class Task():

    """ Manages the creation of tasks by combining data and model cross-entropy, this is where synthetic data is generated. The general Task class contains useful methods for generating whatever tasks you subclass it to. """

    def __init__(
            self, 
            language_model: M, 
            ):
        self.M = language_model

    # @property
    # def xent_call_toks(self):
    #     return self.M.tokenize(self.xent_call_text, padding="do_not_pad").input_ids.to(device)
        
    def random_slice(self, T, length):
        if T.shape[1] <= length:
            return T
        else:
            start = random.randint(0, T.size(1) - length)
            output = T[:, start:start+length]
            return output

    def find_xstring(self, source, target, return_len=False):
        """ Returns the index at which the xent function starts, needed for starting the loss computation """
        xdefseq = self.M.tokenize(target).input_ids.to(device) if isinstance(target, str) else target
        source = self.M.tokenize(source).input_ids.to(device) if isinstance(source, str) else source
        seq_len = xdefseq.shape[1]
        windows = source.unfold(dimension=1, size=seq_len, step=1)
        matches = (windows==xdefseq).all(dim=2)
        indices = matches.nonzero().squeeze(0)
        if return_len:
            return indices[1], seq_len
        return indices[1]
    
    def dataset_generator(
            self, 
            get_sample: Callable,
            out_type: Literal["text", "tokens"],
            **kwargs
        ):
        def iterator(n_samples):
            tracker = tqdm(total=n_samples, desc="samples", disable=True)
            n = 0
            while n < n_samples:
                tok = self.generate(get_sample,**kwargs) # generate should always move in tokens space
                if tok.shape[1] <= self.M.ctx_window:
                    n += 1
                    tracker.update(1)
                    if out_type == "text": yield self.M.detokenize(tok, mode="single") # mode may be wrong but I'll correct it once i'll need to use this
                    elif out_type == "tokens": 
                        yield tok
                    else: raise ValueError("out_type should be 'text' or 'tokens'")
                else: 
                    continue
        return iterator

    def synthesize(
            self, 
            get_sample: Callable,
            n_samples: int,
            out_type: Literal["text", "tokens"],
            **kwargs
        ) -> list[str] | torch.Tensor:
        output = []
        generator = self.dataset_generator(get_sample=get_sample, out_type=out_type, **kwargs)
        for sample in generator(n_samples):
            output.append(sample)
        if out_type == "text": return output # returns List[str]
        elif out_type == "tokens": return torch.cat(output) # returns torch.Tensor


    def dataset_synthesizer(
            self,
            get_sample: Callable, 
            n_samples: int, 
            out_type: Literal["text", "tokens"],
            **kwargs
        ) -> Callable:
        return lambda: self.synthesize(
            get_sample,
            n_samples,
            out_type,
            **kwargs
        )




#######################################################################################################
################################## GENERAL DATAPROCESSING #############################################
#######################################################################################################


class DataProcessor:
        
    def train_test_split(self, train_split = None):
        """ Define a ratio to randomly split the dataset """
        train_size = int(train_split * self.n_samples)
        test_size = self.n_samples - train_size
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])
    
    def exact_split(self, position):
        """ Cut the dataset in a specific way """
        self.train_set = self.dataset[:position]
        self.test_set = self.dataset[position:]

    def get_random_article(self):
        return random.choice(self.dataset)

    def get_random_article_text(self):
        return random.choice(self.dataset)["text"]

    def get_random_train_text(self):
        return random.choice(self.train_set)["text"]

    def get_random_test_text(self):
        return random.choice(self.test_set)["text"]
    
    @classmethod
    def pickle_dump(self, data, task_name, data_name, save_info=None):
        task_dir = os.path.join(data_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        save_path = os.path.join(task_dir, f"{data_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        if save_info: self.save_info_json(save_info, task_dir)
    
    @classmethod
    def save_info_json(self, save_info:dict, path:str):
        save_path = os.path.join(path, "info.json")
        json.dump(save_info, open(save_path, "w+"), indent=4)


#############################################
########## HELPER DATAPROCESSING #############################################################
#############################################

class TokensDataset(Dataset):
    def __init__(self, dataset: torch.Tensor):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]

class TrainTestProcessor(Dataset):
    def __init__(self, train_set, test_set):
        self.train_set, self.test_set = train_set, test_set
    def get_token_loaders(self):
        return TokensDataset(self.train_set), TokensDataset(self.test_set)