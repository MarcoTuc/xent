import os
import json
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
            shuffle=False, 
            log_interval:int=10,
            eval_size:int=100,
            make_samples:bool=True,
            sample_interval=None,
            grad_clip:float=1.0,
            epoch=0,
            ):
        
        # initialize model and data
        self.M = initial_model
        self.D = synthset
        
        # define data-processing relevant parameters
        self.batch_size = batch_size
        self.log_interval = log_interval
        if sample_interval == None: self.sample_interval = log_interval
        else: self.sample_interval = sample_interval
        # we'll load a number of eval_size when doing evaluation
        self.eval_size = min(eval_size, len(self.D.test_set)) 
        # tokenize the dataset if it is made of text. You should flag this in the info.json of the data you generate.
        if self.D._info["data_content"] == "text":
            tqdm.write("Tokenizing the training set:\n")
            self.D.train_set = self.tokenize_dataset(self.D.train_set)
            tqdm.write("Tokenizing the test set:\n")
            self.D.test_set = self.tokenize_dataset(self.D.test_set)
        # make the actual split
        self.train_set, self.test_set = self.D.get_token_loaders()
        # pick batch sized samples and feeds them to the training and evaluation loop
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=shuffle)
        self.gen_loader = DataLoader(self.test_set, batch_size=1, shuffle=True) # pick an example from test set and run generation on it as an input
        
        # trainer options defined externally. CrossEntropy is standard so it's just here. 
        self.crossentropy = CrossEntropyLoss()
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.scheduler = scheduler #TODO implement
        
        # track loss for wandb reporting and model-saving purposes
        self.empty_lossess = torch.tensor([]).to(device)
        self.best_loss = float("inf")
        self.epoch = epoch

        # report the generated samples on wandb
        if make_samples:
            self.gen_table = []
            self.make_samples = make_samples
        
        self.training_steps = len(self.train_set) / self.batch_size
        self.testing_steps = self.eval_size / self.batch_size

    def simple_train(self):
        """ No in-training evaluation of the model and production of a sample at each log interval """
        self.M.model.train()
        losses = self.empty_lossess
        for batch, tokens in enumerate(self.train_loader):
            self.optimizer.zero_grad()           
            logits = self.M.model(input_ids=tokens).logits
            loss = self.compute_batch_loss(logits, tokens)
            if loss == None: continue
            loss.backward()
            clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
            self.optimizer.step()
            losses = torch.cat([losses, loss.unsqueeze(0)])
            if batch % self.log_interval == 0:
                avg_loss = losses.mean().item()
                # wandb.log({"avg_loss": avg_loss})
                if self.make_samples: 
                    prompt, gen_sample = self.gen_in_loop(split=True)
                    self.gen_table.append([avg_loss, prompt, gen_sample])
                    wandb.log({"generated_samples": wandb.Table(
                                        columns=["loss", "prompt", "output"],
                                        data=self.gen_table,
                                        allow_mixed_types=True
                                    )
                                }
                            )
                losses = self.empty_lossess
    
    def train_with_validation(self, saving_options=None, tot_epochs=None):
        self.M.model.train()
        sampling_loss = self.empty_lossess
        total_loss = self.empty_lossess
        for batch, tokens in tqdm(enumerate(self.train_loader), desc=f"Training epoch {self.epoch+1}/{tot_epochs} || ", total=self.training_steps):
            self.optimizer.zero_grad()        
            tokens = tokens.to(device)   
            logits = self.M.model(input_ids=tokens).logits
            loss = self.compute_batch_loss(logits, tokens)
            if loss == None: continue
            loss.backward()
            clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
            self.optimizer.step()
            sampling_loss = torch.cat([sampling_loss, loss.unsqueeze(0)])
            total_loss = torch.cat([total_loss, loss.unsqueeze(0)])
            if self.make_samples and batch % self.sample_interval == 0:
                avg_sample_loss = sampling_loss.mean().item()
                prompt, gen_sample = self.gen_in_loop(split=True)
                self.gen_table.append([avg_sample_loss, prompt, gen_sample])
                sampling_loss = self.empty_lossess
                wandb.log({"generated_samples": wandb.Table(
                                    columns=["loss", "prompt", "output"],
                                    data=self.gen_table,
                                    allow_mixed_types=True
                                )})
            if batch % self.log_interval == 0 and batch > 0: # use log interval as validation interval here
                avg_loss = total_loss.mean().item()
                wandb.log({"train_loss": avg_loss}) # log the train_loss
                self.evaluate(saving_options=saving_options) # will also log the validation loss
                total_loss = self.empty_lossess

    def evaluate(self, saving_options=None):
        tqdm.write("Evaluating the model... ")
        self.M.model.eval()
        valloss = self.empty_lossess
        with torch.no_grad():
            for batch, tokens in tqdm(enumerate(self.test_loader), desc="Testing batch || ", total=self.testing_steps):
                tokens = tokens.to(device)
                logits = self.M.model(input_ids=tokens).logits
                loss = self.compute_batch_loss(logits, tokens)
                if loss == None: continue
                valloss = torch.cat([valloss, loss.unsqueeze(0)])
                if batch == self.testing_steps:
                    break
            val_loss = valloss.mean().item()
            wandb.log({"validation_loss": val_loss})
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if saving_options:
                    self.save_model(**saving_options)
        self.M.model.train()

    def gen_in_loop(self, split=False):
        self.M.model.eval()
        sample = next(iter(self.gen_loader)).to(device)
        xidx, xlen = self.find_xstring(sample, X.xreturn, return_len=True)
        xstart = xidx[1] + xlen
        prompt = sample[0, :xstart]
        attn_mask = torch.ones_like(prompt)
        with torch.no_grad():
            gen = self.M.model.generate(
                prompt.unsqueeze(0), #TODO understand why I need this unsqueeze here to make it work
                attention_mask=attn_mask.unsqueeze(0), #TODO understand why I need this unsqueeze here to make it work
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.M.model.config.eos_token_id,
                max_length=self.M.ctx_window
            )
        self.M.model.train()        
        if split:
            output = self.M.detokenize(gen[0, len(prompt):], mode="tensor")
            prompt = self.M.detokenize(prompt, "tensor")
            return prompt, output
        else:
            return self.M.detokenize(gen[0], mode="tensor")

    def compute_batch_loss(
            self, 
            logits, # [B, T, V]
            tokens, # [B, T]
            ):
        loss = 0
        try: xidx, xlen = self.find_xstring(tokens, X.xreturn, return_len=True)
        except Exception as e: 
            tqdm.write(f"Error in the data, skipping... \n{e}")
            return None # then skip this in the main loop
        for sample, fstart in xidx:
            shift = 0 # KEEP 0 --- change for debugging purposes
            xstart = fstart + xlen - shift
            sample_logits = logits[sample, xstart:-1].view(-1, logits.size(-1)) # [T, V]
            sample_tokens = tokens[sample, xstart+1:].view(-1).long() # [T]
            loss += self.crossentropy(sample_logits, sample_tokens)
        batch_loss = loss / logits.shape[0] # don't use default batch size here
        # wandb.log({"batch_loss": batch_loss})
        return batch_loss # don't use default batch size here

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

    def update_epoch(self, x):
        self.epoch += x
    
    
    ################################
    ######### LOAD / SAVE FACILITIES

    def save_model(self, base=None, model_name=None, new_version=None):
        if base == None: 
            base = self.D.dataset_task 
        if model_name == None: 
            model_name = self.M.model_name
        if new_version == self.M.model_version:
            new_version = f"{new_version}+1"
            tqdm.write(f"New version is the same as old version, changing name to: {new_version}")
        save_path = os.path.join(models_dir, base, model_name, new_version)
        model_save_path = os.path.join(save_path, new_version)
        tqdm.write(f"Saving new model into: {model_save_path}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.M.model, model_save_path)
        self.save_info(save_path)

    def save_info(self, path):
        save = {
            "model_trained_from": {
                "base": self.M.base,
                "name": self.M.model_name,
                "version": self.M.model_version
            },
            "model_trained_on": {
                "task": self.D.task_name,
                "data": self.D.data_name,
                "cut_dataset": self.D.cut_dataset
            },
            "training_status": {
                "validation_loss": self.best_loss,
                "training_epoch": self.epoch,
            }
        }
        print(f"Validation loss: {self.best_loss:.3f} || Epoch: {self.epoch}")
        save_path = os.path.join(path, "info.json")
        json.dump(save, open(save_path, "w+"), indent=4)



class Evolver():
    def __init__(
            self,
            initial_model: M,
            optimizer,
            scheduler,
        ):
        
        self.M = initial_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.best_loss = float("inf")
        self.empty_lossess = torch.tensor([]).to(device)
        self.crossentropy = CrossEntropyLoss()
        self.grad_clip = 1.0


    def train_batch(self, tokens: torch.Tensor, step=None):
        """ Train a batch and return the sum of the lossess per sample """
        self.M.model.train()
        batch_loss = self.empty_lossess
        self.optimizer.zero_grad()        
        tokens = tokens.to(device)   
        logits = self.M.model(input_ids=tokens).logits
        loss = self.compute_batch_loss(logits, tokens)
        if loss == None: return None
        loss.backward()
        clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        batch_loss = torch.cat([batch_loss, loss.unsqueeze(0)])
        return batch_loss
    
    def eval_batch(self, tokens: torch.Tensor):
        """ Evaluate a batch and return the sum of the lossess per sample """
        self.M.model.eval()
        valid_loss = self.empty_lossess
        with torch.no_grad():
            tokens = tokens.to(device)
            logits = self.M.model(input_ids=tokens).logits
            loss = self.compute_batch_loss(logits, tokens)
            if loss == None: return None
            valid_loss = torch.cat([valid_loss, loss.unsqueeze(0)])
        return valid_loss

    def perform_task(self, corpus_sample, split=False):
        self.M.model.eval()
        xidx, xlen = self.find_xstring(corpus_sample, X.xreturn, return_len=True)
        xstart = xidx[1] + xlen
        prompt = corpus_sample[0, :xstart]
        attn_mask = torch.ones_like(prompt)
        with torch.no_grad():
            gen = self.M.model.generate(
                prompt.unsqueeze(0), #TODO understand why I need this unsqueeze here to make it work
                attention_mask=attn_mask.unsqueeze(0), #TODO understand why I need this unsqueeze here to make it work
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.M.model.config.eos_token_id,
                max_length=self.M.ctx_window
            )
        if split:
            output = self.M.detokenize(gen[0, len(prompt):], mode="tensor")
            prompt = self.M.detokenize(prompt, "tensor")
            return prompt, output
        else:
            return self.M.detokenize(gen[0], mode="tensor")

    def compute_batch_loss(
            self, 
            logits, # [B, T, V]
            tokens, # [B, T]
            ):
        loss = 0
        try: xidx, xlen = self.find_xstring(tokens, X.xreturn, return_len=True)
        except Exception as e: 
            tqdm.write(f"Error in the data, skipping... \n{e}")
            return None # then skip this in the main loop
        for sample, fstart in xidx:
            shift = 0 # KEEP 0 --- change for debugging purposes
            xstart = fstart + xlen - shift
            sample_logits = logits[sample, xstart:-1].view(-1, logits.size(-1)) # [T, V]
            sample_tokens = tokens[sample, xstart+1:].view(-1).long() # [T]
            loss += self.crossentropy(sample_logits, sample_tokens)
        batch_loss = loss / logits.shape[0] # don't use default batch size here
        return batch_loss # don't use default batch size here

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

    def save_model(self, base=None, model_name=None, new_version=None):
        if base == None: 
            base = self.D.dataset_task 
        if model_name == None: 
            model_name = self.M.model_name
        if new_version == self.M.model_version:
            new_version = f"{new_version}+1"
            tqdm.write(f"New version is the same as old version, changing name to: {new_version}")
        save_path = os.path.join(models_dir, base, model_name, new_version)
        model_save_path = os.path.join(save_path, new_version)
        tqdm.write(f"Saving new model into: {model_save_path}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.M.model, model_save_path)