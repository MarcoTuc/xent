from config import * 
from models import Model
from xent.dataprocessing import SynthProcessor

class Trainer():

    """ Train the model on a synthetic dataset """

    def __init__(
            self, 
            model: Model,
            synthset: SynthProcessor
            ):

        self.model = model
        self.synthset = synthset
