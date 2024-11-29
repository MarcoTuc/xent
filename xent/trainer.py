import os
import json
from tqdm import tqdm

import wandb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from xent.config import * 
from xent import M, X
from xent.datasets import SynthProcessor

class Trainer():

    """ Train the model on a synthetic dataset """

    def __init__(
            self, 
            initial_model: M,
            train_set, 
            test_set,
            optimizer=None,
            batch_size=None,
            scheduler=None,
            shuffle=False, 
            log_interval:int=10,
            eval_size:int=100,
            make_samples:bool=True,
            sample_interval=None,
            grad_clip:float=1.0,
            report_wandb=True,
            epoch=0,
            ):
        
        # initialize model and data
        self.M = initial_model
        self.train_set, self.test_set = train_set, test_set
        
        # define data-processing relevant parameters
        self.batch_size = batch_size
        self.log_interval = log_interval
        if sample_interval == None: self.sample_interval = log_interval
        else: self.sample_interval = sample_interval
        # we'll load a number of eval_size when doing evaluation
        self.eval_size = min(eval_size, len(self.test_set)) 
        
        # pick batch sized samples and feeds them to the training and evaluation loop
        self.train_loader = DataLoader(
                                self.train_set, 
                                batch_size=self.batch_size, 
                                shuffle=shuffle,
                                pin_memory=True,
                                persistent_workers=True,
                                num_workers=1)
        self.test_loader = DataLoader(
                                self.test_set, 
                                batch_size=self.batch_size, 
                                shuffle=shuffle,
                                pin_memory=True,
                                persistent_workers=True,
                                num_workers=1)
        self.gen_loader = DataLoader(
                                self.test_set, 
                                batch_size=1, 
                                shuffle=True,
                                pin_memory=True,
                                persistent_workers=True,
                                num_workers=1) # pick an example from test set and run generation on it as an input
        
        # trainer options defined externally. CrossEntropy is standard so it's just here. 
        self.crossentropy = CrossEntropyLoss()
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        if scheduler: 
            self.scheduler = scheduler 
            self._do_schedule = True
        else: 
            self._do_schedule = False
        
        self.wandb = report_wandb
        
        # track loss for wandb reporting and model-saving purposes
        self.empty_lossess = torch.tensor([]).to(device)
        self.best_loss = float("inf")
        self.epoch = epoch

        self.train_checkpoint = 0 

        # report the generated samples on wandb
        if make_samples:
            self.gen_table = []
            self.make_samples = make_samples
        
        self.training_steps = len(self.train_set) / self.batch_size
        self.testing_steps = self.eval_size / self.batch_size

    def simple_train(self):
        """ No in-training evaluation of the model and production of a sample at each log interval """
        self.M.model.train()
        sampling_loss = self.empty_lossess
        losses = self.empty_lossess
        for batch, tokens in enumerate(self.train_loader):
            self.optimizer.zero_grad()           
            logits = self.M.model(input_ids=tokens).logits
            loss = self.compute_task_loss(logits, tokens)
            if loss == None: continue
            loss.backward()
            clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self._do_schedule: 
                wandb.log({"learning_rate": self.scheduler.get_last_lr()[0]})
                self.scheduler.step()
            losses = torch.cat([losses, loss.unsqueeze(0)])
            if batch % self.log_interval == 0:
                avg_loss = losses.mean().item()
                if self.make_samples: 
                    prompt, gen_sample = self.gen_in_loop(split=True)
                    self.gen_table.append([avg_loss, prompt, gen_sample])
                    if self.wandb: wandb.log({"generated_samples": wandb.Table(
                                        columns=["loss", "prompt", "output"],
                                        data=self.gen_table,
                                        allow_mixed_types=True
                                    )
                                }
                            )
                losses = self.empty_lossess
    
    def train_task(self, saving_options=None, tot_epochs=None, saving_info=None):
        """ Trains on a xent task and perform validation as well """
        self.M.model.train()
        sampling_loss = self.empty_lossess
        total_loss = self.empty_lossess
        for batch, tokens in tqdm(enumerate(self.train_loader), desc=f"Training epoch {self.epoch+1}/{tot_epochs} || ", total=self.training_steps):
            self.optimizer.zero_grad()        
            tokens = tokens.to(device)   
            logits = self.M.model(input_ids=tokens).logits
            loss = self.compute_task_loss(logits, tokens)
            if loss == None: continue
            loss.backward()
            clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self._do_schedule: 
                wandb.log({"learning_rate": self.scheduler.get_last_lr()[0]})
                self.scheduler.step()
            sampling_loss = torch.cat([sampling_loss, loss.unsqueeze(0)])
            total_loss = torch.cat([total_loss, loss.unsqueeze(0)])
            if self.make_samples and batch % self.sample_interval == 0:
                avg_sample_loss = sampling_loss.mean().item()
                prompt, gen_sample, true = self.gen_in_loop(split=True)
                self.gen_table.append([avg_sample_loss, prompt, gen_sample, true])
                sampling_loss = self.empty_lossess
                if self.wandb: wandb.log({"generated_samples": wandb.Table(
                                    columns=["loss", "prompt", "output", "true"],
                                    data=self.gen_table,
                                    allow_mixed_types=True
                                )})
            if batch % self.log_interval == 0: # use log interval as validation interval here
                avg_loss = total_loss.mean().item()
                if self.wandb: wandb.log({"train_loss": avg_loss}) # log the train_loss
                self.evaluate(saving_options=saving_options, saving_info=saving_info) # will also log the validation loss
                total_loss = self.empty_lossess
            self.train_checkpoint += 1
    
    def pre_train(self, saving_options=None, tot_epochs=None, saving_info=None):
        """ Trains on a xent task and perform validation as well """
        self.M.model.train()
        sampling_loss = self.empty_lossess
        total_loss = self.empty_lossess
        for batch, tokens in tqdm(enumerate(self.train_loader), desc=f"Training epoch {self.epoch+1}/{tot_epochs} || ", total=self.training_steps):
            self.optimizer.zero_grad()        
            tokens = tokens.to(device)   
            logits = self.M.model(input_ids=tokens).logits
            loss = self.crossentropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
            loss.backward()
            clip_grad_norm_(self.M.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self._do_schedule: 
                wandb.log({"learning_rate": self.scheduler.get_last_lr()[0]})
                self.scheduler.step()
            sampling_loss = torch.cat([sampling_loss, loss.unsqueeze(0)])
            total_loss = torch.cat([total_loss, loss.unsqueeze(0)])
            if self.make_samples and batch % self.sample_interval == 0:
                avg_sample_loss = sampling_loss.mean().item()
                prompt, gen_sample, true = self.gen_in_loop(split=True)
                self.gen_table.append([avg_sample_loss, prompt, gen_sample, true])
                sampling_loss = self.empty_lossess
                if self.wandb: wandb.log({"generated_samples": wandb.Table(
                                    columns=["loss", "prompt", "output", "true"],
                                    data=self.gen_table,
                                    allow_mixed_types=True
                                )})
            if batch % self.log_interval == 0: # use log interval as validation interval here
                avg_loss = total_loss.mean().item()
                if self.wandb: wandb.log({"train_loss": avg_loss}) # log the train_loss
                self.evaluate(saving_options=saving_options, saving_info=saving_info) # will also log the validation loss
                total_loss = self.empty_lossess
            self.train_checkpoint += 1

    def evaluate(self, saving_options=None, saving_info=None):
        tqdm.write("Evaluating the model... ")
        self.M.model.eval()
        valloss = self.empty_lossess
        with torch.no_grad():
            for batch, tokens in tqdm(enumerate(self.test_loader), desc="Testing batch || ", total=self.testing_steps):
                tokens = tokens.to(device)
                logits = self.M.model(input_ids=tokens).logits
                loss = self.compute_task_loss(logits, tokens)
                if loss == None: continue
                valloss = torch.cat([valloss, loss.unsqueeze(0)])
                if batch == self.testing_steps:
                    break
            val_loss = valloss.mean().item()
            if self.wandb: wandb.log({"validation_loss": val_loss})
            else: print("validation loss:", val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if saving_options:
                    self.save_model(**saving_options, saving_info=saving_info)
        self.M.model.train()

    def gen_in_loop(self, split=False):
        self.M.model.eval()
        sample = next(iter(self.gen_loader)).to(device)
        xidx, xlen = self.find_xstring(sample, X.xreturn, return_len=True)
        # print(sample)
        xstart = xidx[1] + xlen + 1 # the +1 is needed to append the \n character without with the model gets confused (...)
        prompt = sample[:, :xstart]
        true = sample[:, xstart:]
        # print(true)
        attn_mask = torch.ones_like(prompt)
        with torch.no_grad():
            gen = self.M.model.generate(
                prompt, #TODO understand why I need this unsqueeze here to make it work
                attention_mask=attn_mask, #TODO understand why I need this unsqueeze here to make it work
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.M.model.config.eos_token_id,
                max_length=self.M.ctx_window
            )
        self.M.model.train()        
        if split:
            output = self.M.detokenize(gen[0, prompt.shape[1]:], mode="tensor")
            prompt = self.M.detokenize(prompt[0], "tensor")
            true = self.M.detokenize(true[0], "tensor")
            return prompt, output, true
        else:
            return self.M.detokenize(gen[0], mode="tensor")

    def compute_task_loss(
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
            xstart = fstart + xlen + shift
            sample_logits = logits[sample, xstart:-1].view(-1, logits.size(-1)) # [T, V]
            sample_tokens = tokens[sample, xstart+1:].view(-1).long() # [T]
            loss += self.crossentropy(sample_logits, sample_tokens)
        batch_loss = loss / logits.shape[0] # batch size at logits.shape[0]
        return batch_loss 

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

    def save_model(self, base=None, model_name=None, new_version=None, saving_info=None):
        if base == None: 
            raise Exception("dude provide a base in a dictionary to pass as saving_options")
        if model_name == None: 
            raise Exception("dude provide a model_name in a dictionary to pass as saving_options")
        if new_version == self.M.model_version:
            raise Exception("dude provide a model_version  in a dictionary to pass as saving_options")
        save_path = os.path.join(models_dir, base, model_name, new_version)
        model_save_path = os.path.join(save_path, new_version)
        tqdm.write(f"Saving new model into: {model_save_path}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.M.model, model_save_path)
        self.save_info(save_path, saving_info)

    def save_info(self, path, saving_info):
        save = {
            "model_trained_from": {
                "base": self.M.base,
                "name": self.M.model_name,
                "version": self.M.model_version
            },
            "saving_info": saving_info,
            "training_status": {
                "validation_loss": self.best_loss,
                "training_epoch": self.epoch,
                "training_checkpoint": self.train_checkpoint,
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