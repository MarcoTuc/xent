import torch
import torch.nn.functional as F

from transformers import Phi3ForCausalLM

PADDING = True 

class LLM:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

class Synth:
    def __init__(self, llm):
        self.llm = llm

    def phi_suggestions(self, sys_prompt):
        if not isinstance(self.llm, Phi3ForCausalLM):
            raise AssertionError("Need to instantiate a Phi3ForCausalLM model")


class CrossEntropyDifferential:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.tokenized_y = None
        self.cross_y = None

    def __call__(self, X, Y, diff=True) -> torch.Tensor:
        """ Compute the cross entropy of sentence Y given sentence X """
    
        len_x = len(self.tokenizer(X, return_tensors='pt', padding=PADDING).input_ids[0])
        zt = self.tokenizer(X + Y, return_tensors='pt', padding=PADDING).to(self.device)
        
        if diff: 
            if self.cross_y is None:
                self.cross_y = self.get_std_cross_entropy(Y)
            with torch.no_grad():
                output_z = self.model(**zt)
            logits_z = output_z.logits
            cross_z = F.cross_entropy(logits_z[0, len_x:][:-1], zt.input_ids[0, len_x:][1:])
            return cross_z - self.cross_y
        
        else:
            with torch.no_grad():
                output = self.model(**zt)
            logits = output.logits
            return F.cross_entropy(logits[0, len_x:][:-1], zt.input_ids[0, len_x:][1:])

    def get_std_cross_entropy(self, Y):
        self.tokenized_y = self.tokenizer(Y, return_tensors='pt', padding=PADDING).to(self.device)
        with torch.no_grad():   
            output_y = self.model(**self.tokenized_y)
        logits_y = output_y.logits
        return F.cross_entropy(logits_y[0][:-1], self.tokenized_y.input_ids[0][1:])