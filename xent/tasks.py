from typing import Callable

import torch 
import torch.nn.functional as F
from xent import Task, X, M, device

class Closure(Task):
    
    # the xent-function string represnting the task, will automatically be tokenize by the task class
    xent_call_text = f"\n{X.xdef} closure{X.opent}{X.clost}{X.xreturn}\n"
#   xent_call_toks = corresponding tokenized string made by Task parent class if you want    
    
    def __init__(
            self, 
            language_model: M,
            inverse_order: bool = False
            ):
        super().__init__(language_model=language_model)
        
        self.inverse_order = inverse_order

        if inverse_order: 
            self.xent_call_text = f"\n{X.xdef} inverseclosure{X.opent}{X.clost}{X.xreturn}\n"
        else: 
            self.xent_call_text = f"\n{X.xdef} closure{X.opent}{X.clost}{X.xreturn}\n"
        
        self.xent_call_toks = self.M.tokenize(self.xent_call_text).input_ids.to(device)
        self.colon = self.M.tokenize(":").input_ids.to(device)
        self.semicolon = self.M.tokenize(";").input_ids.to(device)
        self.newline = self.M.tokenize("\n").input_ids.to(device)

        # Instantiate xent functions for the parallel_parallel data generator

        self.fwd_closure_toks_i = self.M.tokenize(
            X.xmap("\ndef fwd_closure(integer):\n")
        ).input_ids.to(device)
        self.fwd_closure_toks_f = self.M.tokenize(
            X.xmap("\ndef fwd_closure(floor):\n")
        ).input_ids.to(device)
        self.bwd_closure_toks_i = self.M.tokenize(
            X.xmap("\ndef bwd_closure(integer):\n")
        ).input_ids.to(device)
        self.bwd_closure_toks_f = self.M.tokenize(
            X.xmap("\ndef bwd_closure(floor):\n")
        ).input_ids.to(device)
        self.xonly_fwd_toks_i = self.M.tokenize(
            X.xmap("\ndef xonly_fwd(integer):\n")
        ).input_ids.to(device)
        self.xonly_fwd_toks_f = self.M.tokenize(
            X.xmap("\ndef xonly_fwd(floor):\n")
        ).input_ids.to(device)
        self.xonly_bwd_toks_i = self.M.tokenize(
            X.xmap("\ndef xonly_bwd(integer):\n")
        ).input_ids.to(device)
        self.xonly_bwd_toks_f = self.M.tokenize(
            X.xmap("\ndef xonly_bwd(floor):\n")
        ).input_ids.to(device)

        self.rank_task = self.M.tokenize(
            X.xmap("\ndef xent_rank():\n")
        ).input_ids.to(device)


    def generate_parallel_parallel(
            self,
            get_sample: Callable,
        ):

        """ This is a special generator for the parallel_parallel generator, see message string there """
        
        share = int(1/5.1 * self.M.ctx_window)
        rescaling = 4

        corpus = get_sample() # get corpus of text from callable method
        toks = self.M.tokenize(corpus).input_ids # get tokenized corpus
        window = min(share, toks.shape[1])
        toks = toks.unfold(1, window, window).squeeze(0)
        
        out_fwd_closure_i = torch.LongTensor([]).to(device)
        out_fwd_closure_f = torch.LongTensor([]).to(device)
        out_xonly_fwd_i = torch.LongTensor([]).to(device)
        out_xonly_fwd_f = torch.LongTensor([]).to(device)
        out_bwd_closure_i = torch.LongTensor([]).to(device)
        out_bwd_closure_f = torch.LongTensor([]).to(device)
        out_xonly_bwd_i = torch.LongTensor([]).to(device)
        out_xonly_bwd_f = torch.LongTensor([]).to(device)

        out_xent_rank = torch.LongTensor([]).to(device)
        
        for slice in toks:

            slice = slice.unsqueeze(0)
            logits = self.M.model(slice).logits
            xent = F.cross_entropy(logits[:, :-1].view(-1, logits.shape[-1]), slice[:, 1:].view(-1), reduction="none")

            # generate forward closure tasks
            fwd_closure_toks_i = torch.cat([slice, self.fwd_closure_toks_i], dim=-1)
            fwd_closure_toks_f = torch.cat([slice, self.fwd_closure_toks_f], dim=-1)
            xonly_fwd_toks_i = torch.cat([slice, self.xonly_fwd_toks_i], dim=-1)
            xonly_fwd_toks_f = torch.cat([slice, self.xonly_fwd_toks_f], dim=-1)
            for tok, xnt in zip(slice[0, 1:], xent):
                xentok_i = self.M.tokenize(f" {round(float(xnt))}").input_ids.to(device)
                xentok_f = self.M.tokenize(f" {round(float(xnt)/rescaling)}").input_ids.to(device)
                fwd_closure_toks_i = torch.cat([fwd_closure_toks_i, tok.view(1,1), self.colon, xentok_i, self.newline], dim=-1)
                fwd_closure_toks_f = torch.cat([fwd_closure_toks_f, tok.view(1,1), self.colon, xentok_f, self.newline], dim=-1)
                xonly_fwd_toks_i = torch.cat([xonly_fwd_toks_i, xentok_i, self.semicolon], dim=-1)
                xonly_fwd_toks_f = torch.cat([xonly_fwd_toks_f, xentok_f, self.semicolon], dim=-1)
            
            # generate backward closure tasks
            bwd_closure_toks_i = torch.cat([slice, self.bwd_closure_toks_i], dim=-1)
            bwd_closure_toks_f = torch.cat([slice, self.bwd_closure_toks_f], dim=-1)
            xonly_bwd_toks_i = torch.cat([slice, self.xonly_bwd_toks_i], dim=-1)
            xonly_bwd_toks_f = torch.cat([slice, self.xonly_bwd_toks_f], dim=-1)
            for tok, xnt in zip(slice[0, 1:].flip(0), xent.flip(0)):
                xentok_i = self.M.tokenize(f" {round(float(xnt))}").input_ids.to(device)
                xentok_f = self.M.tokenize(f" {round(float(xnt)/rescaling)}").input_ids.to(device)
                bwd_closure_toks_i = torch.cat([bwd_closure_toks_i, tok.view(1,1), self.colon, xentok_i, self.newline], dim=-1)
                bwd_closure_toks_f = torch.cat([bwd_closure_toks_f, tok.view(1,1), self.colon, xentok_f, self.newline], dim=-1)
                xonly_bwd_toks_i = torch.cat([xonly_bwd_toks_i, xentok_i, self.semicolon], dim=-1)
                xonly_bwd_toks_f = torch.cat([xonly_bwd_toks_f, xentok_f, self.semicolon], dim=-1)

            # generate ranking tasks
            xent_rank_toks = torch.cat([slice, self.rank_task], dim=-1)
            ranked_xents, sorting = torch.sort(xent, descending=True)
            sorted_toks = toks[0, 1:][sorting]
            task_toks = torch.cat([
                sorted_toks.view(sorted_toks.shape[0], 1),
                # torch.cat([self.M.tokenize(f" {round(float(x)/rescaling)}").input_ids for x in ranked_xents]), # uncomment if you also want to put xent in the output task
                self.semicolon.squeeze(0).repeat(sorted_toks.shape[0], 1)
            ], dim = 1)
            xent_rank_toks = torch.cat([xent_rank_toks, task_toks.view(1,-1)], dim=-1)

            # stack everything for return
            out_fwd_closure_i = torch.cat([out_fwd_closure_i, self.M.pad(fwd_closure_toks_i)])
            out_fwd_closure_f = torch.cat([out_fwd_closure_f, self.M.pad(fwd_closure_toks_f)])
            out_xonly_fwd_i = torch.cat([out_xonly_fwd_i, self.M.pad(xonly_fwd_toks_i)])
            out_xonly_fwd_f = torch.cat([out_xonly_fwd_f, self.M.pad(xonly_fwd_toks_f)])
            out_bwd_closure_i = torch.cat([out_bwd_closure_i, self.M.pad(bwd_closure_toks_i)])
            out_bwd_closure_f = torch.cat([out_bwd_closure_f, self.M.pad(bwd_closure_toks_f)])
            out_xonly_bwd_i = torch.cat([out_xonly_bwd_i, self.M.pad(xonly_bwd_toks_i)])
            out_xonly_bwd_f = torch.cat([out_xonly_bwd_f, self.M.pad(xonly_bwd_toks_f)])
            out_xent_rank = torch.cat([out_xent_rank, self.M.pad(xent_rank_toks)])

        return {
            "fwd_closure_i": out_fwd_closure_i,
            "fwd_closure_f": out_fwd_closure_f,
            "xonly_fwd_i": out_xonly_fwd_i,
            "xonly_fwd_f": out_xonly_fwd_f,
            "bwd_closure_i": out_bwd_closure_i,
            "bwd_closure_f": out_bwd_closure_f,
            "xonly_bwd_i": out_xonly_bwd_i,
            "xonly_bwd_f": out_xonly_bwd_f,
            "xent_rank_top": out_xent_rank
        }

    def generate(
            self,
            get_sample: Callable,
            preprompt_share=1/5.05,
        ):
        corpus = get_sample() # get corpus of text from callable method
        toks = self.M.tokenize(corpus).input_ids # get tokenized corpus
        sliced_toks = self.random_slice(toks, int(self.M.ctx_window * preprompt_share)) # get a random slice of the tokens
        output_toks = torch.cat([sliced_toks, self.xent_call_toks], dim=-1) # concatenate the xent function to the corpus
        logits = self.M.model(sliced_toks).logits
        xent = F.cross_entropy(logits[:, :-1].view(-1, logits.shape[-1]), sliced_toks[:, 1:].view(-1), reduction="none")
        xent_toks = torch.tensor([], dtype=torch.int).to(device) # initialize an empty tensor to concat xents into
        semicolon = self.M.tokenize(":").input_ids.squeeze(0)
        newline = self.M.tokenize("\n").input_ids.squeeze(0)
        iterator = zip(sliced_toks[0, 1:].flip(0), xent.flip(0)) if self.inverse_order else zip(sliced_toks[0, 1:], xent)
        # loop to add the various {tok: xent} lines 
        for tok, xnt in iterator: # xent[i] is for sliced_toks[i+1] as it should be
            xntok = self.M.tokenize(f" {round(float(xnt))}").input_ids.to(device).squeeze(0)
            xent_toks = torch.cat([xent_toks, tok.unsqueeze(0), semicolon, xntok, newline], dim=-1)
        output_toks = torch.cat([output_toks, xent_toks.unsqueeze(0)], dim=-1)
        return self.M.pad(output_toks) # tokenizer padding is "do_not_pad" so we need to pad things at the end
    
    def generate_multi(
            self,
            get_sample: Callable,
        ):
        share = int(1/5.05 * self.M.ctx_window)
        corpus = get_sample() # get corpus of text from callable method
        toks = self.M.tokenize(corpus).input_ids # get tokenized corpus
        window = min(share, toks.shape[1])
        toks = toks.unfold(1, window, window).squeeze(0)
        semicolon = self.M.tokenize(":").input_ids.squeeze(0)
        newline = self.M.tokenize("\n").input_ids.squeeze(0)
        output_toks = torch.LongTensor([]).to(device)
        for slice in toks:
            slice = slice.unsqueeze(0)
            new_toks = torch.cat([slice, self.xent_call_toks], dim=-1) # concatenate the xent function to the corpus
            logits = self.M.model(slice).logits
            xent = F.cross_entropy(logits[:, :-1].view(-1, logits.shape[-1]), slice[:, 1:].view(-1), reduction="none")
            xent_toks = torch.tensor([], dtype=torch.int).to(device) # initialize an empty tensor to concat xents into
            iterator = zip(slice[0, 1:].flip(0), xent.flip(0)) if self.inverse_order else zip(slice[0, 1:], xent)
            # loop to add the various {tok: xent} lines 
            for tok, xnt in iterator: # xent[i] is for sliced_toks[i+1] as it should be
                xntok = self.M.tokenize(f" {round(float(xnt))}").input_ids.to(device).squeeze(0)
                xent_toks = torch.cat([xent_toks, tok.unsqueeze(0), semicolon, xntok, newline], dim=-1)
            new_toks = torch.cat([new_toks, xent_toks.unsqueeze(0)], dim=-1)
            output_toks = torch.cat([output_toks, self.M.pad(new_toks)]) # tokenizer padding is "do_not_pad" so we need to pad things at the end
        return output_toks
    
    