import sys
from typing import Type, Literal, Callable, Union, List
from tqdm import tqdm
import random

import torch 

from xent.dataprocessing import DataProcessor
from xent.config import * 
from xent.models import M
from xent.lang import X


class Task():

    """ Manages the creation of tasks by combining data and model cross-entropy, this is where synthetic data is generated. The general Task class contains useful methods for generating whatever tasks you subclass it to. """

    def __init__(
            self, 
            language_model: M, 
            ):

        self.M = language_model
    
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
            out_type: Literal["text", "tokens"]
        ):
        def iterator(n_samples):
            tracker = tqdm(total=n_samples, desc="samples", disable=True)
            n = 0
            while n < n_samples:
                new = self.generate(get_sample, space=out_type)
                tok = self.M.tokenize(new).input_ids if out_type == "text" else new
                if tok.shape[1] <= self.M.ctx_window:
                    n += 1
                    tracker.update(1)
                    if out_type == "text": yield new
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
            out_type: Literal["text", "tokens"]
        ):
        output = []
        generator = self.dataset_generator(get_sample=get_sample, out_type=out_type)
        for sample in generator(n_samples):
            output.append(sample)
        if out_type == "text": return output
        elif out_type == "tokens": return torch.cat(output)


    def dataset_synthesizer(
            self,
            get_sample: Callable, 
            n_samples: int, 
            out_type: Literal["text", "tokens"],
        ) -> Callable:
        return lambda: self.synthesize(
            get_sample,
            n_samples,
            out_type
        )


class Closure(Task):

    def __init__(
            self, 
            language_model: M,
            ):
        super().__init__(
            language_model, 
            )
    
        # the xent-function represnting the task
        self.xent_call_text = f"\n{X.xdef} closure{X.opent}{X.clost}{X.xreturn}\n"
        # tokenized - do not pad as you only want to pad the completed task
        self.xent_call_toks = self.M.tokenize(self.xent_call_text, padding="do_not_pad").input_ids.to(device)

    def generate(
            self,
            get_sample: Callable,
            preprompt_share=1/5.05,
            space="tokens"
        ):
        # get corpus of text from callable method
        corpus = get_sample()
        # get tokenized corpus
        toks = self.M.tokenize(corpus).input_ids
        # get a random slice of the tokens
        sliced_toks = self.random_slice(toks, int(self.M.ctx_window * preprompt_share))
        # get the list of cross-entropy of sliced tokens
        xent = self.M.get_xent(sliced_toks)
        # concatenate the xent function to the corpus
        output_toks = torch.cat([sliced_toks, self.xent_call_toks], dim=-1)
        # initialize an empty tensor to concat xents into
        xent_toks = torch.tensor([], dtype=torch.int).to(device)
        # tokenize structural things 
        semicolon = self.M.tokenize(":").input_ids.squeeze(0)
        newline = self.M.tokenize("\n").input_ids.squeeze(0)
        # loop to add the various {tok: xent} lines 
        for tok, xnt in zip(sliced_toks[0, 1:], xent): # xent is already left shifted
            xntok = self.M.tokenize(f" {round(float(xnt))}").input_ids.to(device).squeeze(0)
            xent_toks = torch.cat([xent_toks, tok.unsqueeze(0), semicolon, xntok, newline], dim=-1)
        output_toks = torch.cat([output_toks, xent_toks.unsqueeze(0)], dim=-1)
        return self.M.pad(output_toks) # tokenizer padding is "do_not_pad" so we need to pad things at the end


class Highlight(Task):

    """ TODO: task under construction -- use highlighting folder for highlighting related training loop """
        
    def __init__(
            self, 
            language_model: M, 
            dataprocessor: Type[DataProcessor]):
        super().__init__(language_model, dataprocessor)
    
    def phrasize(self, preprompt):
        """ This method uses the template to create a phrasized version of a keyword argument that is extracted to be used in the preprompt """
        templates =  [f"{t}\n" for t in [
            "This is what I mean when I talk about {}.",
            "{} is a pretty good title for this.",
            "I found out that {} is about the following.",
            "The things most commonly associated with {} are these.",
            "I think this is about {}.",
            "Whatever you say, this is about {}.",
            "And here's what reminds me of {}.",
            "{} screams from every corner of this.",
            "If {} were a flavor, this would taste like it.",
            "This dances to the rhythm of {}.",
            "This has {} written all over it.",
            "You can smell {} all over this.",
            "If {} were a book, this would be its intro.",
            "This is what {} looks like in reality.",
            "This makes {} look like a close relative topic.",
            ]
        ]
        return self.M.tokenize(random.choice(templates).format(self.M.detokenize(preprompt, mode="single"))).input_ids

    def get_wiki_task(self):
        A1 = self.D.get_random_article() # string
        A2 = self.D.get_random_article() # string
        A1, A2 = self.M.tokenize(A1), self.M.tokenize(A2)
        
    def random_slice(self, T, length):
        if T.size(1) <= length:
            return T
        start = random.randint(0, T.size(1) - length)
        return T[:, start:start + length]