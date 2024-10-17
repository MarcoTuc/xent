import os

from time import time
from tqdm import tqdm

import wandb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from xent.config import * 
from xent.lang import X
from xent.models import M
from xent.dataprocessing import SynthProcessor

class Trainer():

    """ Train the model on a synthetic dataset """

    def __init__(
            self, 
            initial_model: M,
            synthset: SynthProcessor,
            optimizer,
            batch_size,
            scheduler=None,
            shuffle=False, # keep false to avoid overlapping after each evolution step
            grad_clip=1.0,
            log_interval=10,
            ):

        self.M = initial_model
        self.D = synthset
        self.batch_size = batch_size
        if self.D._info["data_content"] == "text":
            print("Tokenizing the training set:\n")
            self.D.train_set = self.tokenize_dataset(self.D.train_set)
            print("Tokenizing the test set:\n")
            self.D.test_set = self.tokenize_dataset(self.D.test_set)
        self.train_set, self.test_set = self.D.get_token_loaders()
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=shuffle)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.crossentropy = CrossEntropyLoss()
        self.grad_clip = grad_clip
        self.log_interval = log_interval

    def simple_train(self):
        loss_series = []
        self.M.model.train()
        report_loss = 0
        now = time()
        for batch, tokens in enumerate(self.train_loader):
            self.optimizer.zero_grad()           
            logits = self.M.model(input_ids=tokens).logits
            loss = self.compute_batch_loss(logits, tokens)
            if loss == None: continue
            loss.backward()
            clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
            self.optimizer.step()
            report_loss += loss.detach().data
            if (batch+1) % self.log_interval == 0 and batch > 0:
                cur_loss = report_loss / self.log_interval
                loss_series.append(cur_loss)
                now = time() - now
                print(f"|| batch: {batch+1} | loss: {cur_loss:.3f} | has taken: {now:.3f} seconds")
                now = time()
                report_loss = 0
             
    def compute_batch_loss(
            self, 
            logits, # [B, T, V]
            tokens, # [B, T]
            ):
        loss = 0
        try: xidx, xlen = self.find_xstring(tokens, X.xreturn, return_len=True)
        except Exception as e: 
            print(f"Error in the data, skipping... \n{e}")
            return None # then skip this in the main loop
        for sample, fstart in xidx:
            shift = 0 # KEEP 0 --- change for debugging purposes
            xstart = fstart + xlen - shift
            sample_logits = logits[sample, xstart:-1].view(-1, logits.size(-1)) # [T, V]
            sample_tokens = tokens[sample, xstart+1:].view(-1).long() # [T]
            print(self.M.detokenize(sample_tokens, mode="tensor"))
            loss += self.crossentropy(sample_logits, sample_tokens)
        return loss / logits.shape[0] # don't use default batch size here

    def find_xstring(self, tokens, string, return_len=False):
        #TODO this method exists both in Task() and Trainer() classes. Should make it unique. 
        """ Returns the index at which the xent function starts, needed for starting the loss computation """
        xdefseq = self.M.tokenize(string).input_ids.to(device)
        seq_len = xdefseq.shape[1]
        windows = tokens.unfold(dimension=1, size=seq_len, step=1)
        matches = (windows==xdefseq).all(dim=2)
        indices = matches.nonzero().squeeze(0)
        if return_len:
            return indices, seq_len
        return indices[1]

    def tokenize_dataset(self, data):
        tokenized = [self.M.tokenize(text, padding="max_length").input_ids for text in tqdm(data)]
        return torch.cat(tokenized)
