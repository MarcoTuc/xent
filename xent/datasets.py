### ANCHE IO MI SONO UPPATO


import os
import json
import pickle
from tqdm import tqdm

import torch
from datasets import load_dataset

from xent import DataProcessor
from xent.base import TokensDataset
from xent.config import * 


#############################################
############## CORPUS DATASETS ########################################################################à
#############################################

class Wikipedia(DataProcessor):
    def __init__(self, split=None):
        self.database = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)["train"]
        self.num_articles = len(self.database)
        self.n = 0
        if isinstance(split, float):
            self.train_test_split(split)
        
    def get_next_article(self):
        article = self.database[self.n]["text"]
        self.n += 1
        return article


class SkeinAdventures(DataProcessor):    
    def __init__(self, split=None):
        self.database = load_dataset("ToastyPigeon/skein-text-adventures")["train"]
        self.num_articles = len(self.database)
        if isinstance(split, float):
            self.train_test_split(split)


#######################################################
############# SYNTHETIC DATA PROCESSING ########################################################################à
#######################################################

class SynthProcessor(DataProcessor):

    """ Process a synthetic dataset for training """

    def __init__(
            self,
            base,
            dataset_name,
            train_split=None,
            split_posit=None,
            cut_dataset=None,
            cut_trainset=None,
            files_to_load=None,
            ):
        
        self.base = base
        self.data_name = dataset_name

        self._path = os.path.join(data_dir, base, dataset_name)
        self._info = json.load(open(os.path.join(self._path, "info.json"), "r+"))
        self.train_split = train_split
        self.cut_dataset = cut_dataset
        
        # to cut the whole dataset
        if cut_dataset:
            self.dataset = self.dataset[:cut_dataset]
            self.n_samples = len(self.dataset)
        
        # splitting the dataset
        if train_split: self.train_test_split(train_split)
        if split_posit: self.exact_split(split_posit)
        
        # after splitting, if you want to cut the training set 
        if cut_trainset:
            self.train_set = self.train_set[:cut_trainset]
        
        self.files_to_load = files_to_load

        self.dataset = self.load_pickled_dataset()
    
    
    def load_pickled_dataset(self):
        data_files = [f for f in os.listdir(self._path) if f.endswith(".pkl")]
        if self.files_to_load: data_files = data_files[:self.files_to_load]
        output = []
        for file in tqdm(data_files, desc="Loading data files"): 
            with open(os.path.join(self._path, f"{file}"), "rb") as data:
                if self._info["data_type"] == "tokens": output.append(pickle.load(data).to("cpu")) # safety measure since many guys are initialized to be on the gpu (my bad)
                elif self._info["data_type"] == "list": output.extend(pickle.load(data).to("cpu"))
        if self._info["data_type"] == "tokens": output = torch.cat(output)
        self.n_samples = len(output)
        return output

    def get_token_loaders(self):
        return TokensDataset(self.train_set), TokensDataset(self.test_set)
    

