import random
import pickle
from datasets import load_dataset

from xent.config import * 
from xent.lang import X

class DataProcessor:
    
    data_dir = os.path.join(work_dir, "data")
    
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
            train_split: float
            ):
        
        self.train_split = train_split
