from typing import Type

import random

from dataprocessing import DataProcessor
from config import * 
from models import M
from xentlang import X

class Task():

    """ Manages the creation of tasks by combining data and model cross-entropy, this is where synthetic data is generated """

    def __init__(
            self, 
            language_model: M, 
            datapreprocessor: Type[DataProcessor]
            ):

        self.M = language_model
        self.D = datapreprocessor
    

class Highlight(Task):
        
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