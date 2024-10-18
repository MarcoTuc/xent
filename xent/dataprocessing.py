import os
import json
import random
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset

from xent.config import * 
from xent.utils import * 
from xent.lang import X


class DataProcessor:
        
    @classmethod
    def pickle_dump(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


class Wikipedia(DataProcessor):
    
    def __init__(self):
        self.database = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)["train"]
        self.num_articles = len(self.database)
    
    def get_random_article(self):
        return random.choice(self.database)

    def get_random_article_text(self):
        return random.choice(self.database)["text"]


class SynthProcessor(DataProcessor):

    """ Process a synthetic dataset for training """

    def __init__(
            self,
            dataset_task,
            dataset_name,
            train_split: float,
            cut_dataset=None
            ):
        
        self.task_name = dataset_task
        self.data_name = dataset_name

        self._path = os.path.join(data_dir, dataset_task, dataset_name)
        self._info = json.load(open(os.path.join(self._path, "info.json"), "r+"))
        self.train_split = train_split
        self.dataset = self.load_pickled_dataset()
        self.cut_dataset = cut_dataset
        if cut_dataset:
            self.dataset = self.dataset[:cut_dataset]
            self.n_samples = len(self.dataset)
        self.train_test_split()

    def load_pickled_dataset(self):
        data_files = [f for f in os.listdir(self._path) if f.endswith(".pkl")]
        output = []
        for file in data_files: 
            with open(os.path.join(self._path, f"{file}"), "rb") as data:
                if self._info["data_type"] == "tensor": output.append(pickle.load(data))
                elif self._info["data_type"] == "list": output.extend(pickle.load(data))
        if self._info["data_type"] == "tensor": output = torch.cat(output).to(device)
        self.n_samples = len(output)
        return output

    
    def train_test_split(self, train_split = None):
        if train_split == None:
            split = self.train_split
        train_size = int(split * self.n_samples)
        test_size = self.n_samples - train_size
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])

    def get_token_loaders(self):
        return TokensDataset(self.train_set), TokensDataset(self.test_set)
    

class TokensDataset(Dataset):
    def __init__(self, dataset: torch.Tensor):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]