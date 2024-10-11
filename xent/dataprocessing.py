import random
from datasets import load_dataset

from config import * 
from xentlang import X

class DataProcessor:
    def __init__():
        pass

class Wikipedia(DataProcessor):
    
    def __init__(self):
        self.database = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)["train"]
        self.num_articles = len(self.database)
    
    def get_random_article(self):
        return random.choice(self.database)        


class SynthProcessor(DataProcessor):

    """ Process a synthetic dataset for training """

    def __init__(
            self,
            train_split: float
            ):
        
        self.train_split = train_split
