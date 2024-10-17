### SETUP

1. create an environment inside this folder and call it menv: `python -m venv menv`

2. activate the environment `source menv/bin/activate`

3. Install the needed packages inside the menv environent: `pip install -r requirements.txt`

4. With menv activated, install the xent package as: `python setup.py develop`

5. To populate the models folder with the base models you can use the *model_downloader.ipynb* notebook

NOTE: Make sure you have nvcc>=11.6 on your Nvidia GPU

-----

### The Xent Package

> **dataprocessing.py**: \
Classes and methods to manipulate seed and synthetic datasets for subsequent generation and training purposes.

> **lang.py**: \
String manipulation methods to represent tasks in the xent language (eg: lots of unique symbols for the LLM to learn xent tasks)

>**models.py**: \
Utilities and some abstraction for loading, saving and using LLMs, their tokenizers and important parameters.     It serves two purposes: use it to initialize a Task class to generate data with a model internals or use it to train the model via Trainer class.

>**tasks.py**: \
Classes and methods that abstract the creation of tasks. You can create a new task by subclassing from the Task class and writing custom methods in it. The Task class has some handy methods to help you write down everything. By combining these with the abstractions in the M class, task writing should become easier. Ultimately a Task class is something that should be initialized with a model and used to generate data by combining it with a seed data extractor method which should return text to manipulate inside the task generation methods.

>*under construction* -- **trainer.py**: \
Give it a model and a dataloader and you're training boyy.

>**utils.py**: \
Put general utilities here. For now we have a Tee class that I use to direct stdout to a txt file.

>**config.py**: \
General boring configs like directories but also funny ones like the device you're using. 

--------------------
## How to use it:

### Generating data:
Import the necessary components
```
from xent.tasks import Closure
from xent.models import M
from xent.lang import X
from xent.dataprocessing import Wikipedia
```
You need to define the model you use to generate the data, the source of data corpus and the task you're going to execute on the data to generate new one.
```
model = M("gpt2", "M0")
data = Wikipedia()
task = Closure(model)
```
Then you need to specify a method for retrieving corpus. In this case we're simply retrieving a random portion of text from a random wikipedia article. Then just call the synthesize_dataset method and you're set. You can either produce data and keep it as a tensor with `out_type="tensor"` or you can set it to `"text"` and generate a text dataset. If you are planning to use the same model then the tensor option will be more efficient but if you have to change model (hence tokenizer) it's best ot save the dataset to text.
```
get_corpus_method = data.get_random_article_text
new_points_to_generate = 10
new_data = task.synthesize_dataset(get_corpus_method, new_points_to_generate, out_type="tensor")
```
### Defining a task:
To define a task you need to subclass from the Task class in xent.tasks and provide the task with a language model and a generate method. Then everything else happens with method you inherit from the Task class. 

Here's an example: 
```
class Closure(Task):

    def __init__(
            self, 
            language_model: M,
            ):
        super().__init__(
            language_model, 
            )

    def generate(
            self,
            get_sample: Callable,
        ):
        preprompt_share = 1/5
        original_text = get_sample()
        toks = self.M.tokenize(original_text).input_ids
        sliced_toks = self.random_slice(toks, int(self.M.ctx_window * preprompt_share))
        xent = self.M.get_xent(sliced_toks)
        stok = self.M.detokenize(sliced_toks, mode="list")
        sliced_text = self.M.detokenize(sliced_toks, mode="single")
        output_text = sliced_text + f"\n{X.xdef} closure{X.opent}{X.clost}{X.xreturn}\n"
        for txt, xnt in zip(stok[1:], xent):
            output_text = output_text + f"{txt}: {round(float(xnt))}\n"
        return output_text
```
### Training a model: 
Once you generated a dataset and saved it, you process it through the SynthProcessor which will take care of train/test split and providing the Trainer class with dataloaders and everything. \
Just provide the Trainer with a model, a synth processor and an optimizer and you're set. The simple_train() method will take care of going though one epoch of training. 
```
from torch.optim import AdamW
from xent.dataprocessing import SynthProcessor
from xent.models import M
from xent.trainer import Trainer

model = M("gpt2", "M0")
synthdata = SynthProcessor("closure", "D0", train_split=0.9, cut_dataset=5000)
optimizer = AdamW(model.model.parameters(), lr=6e-4, betas=(0.1, 0.95), weight_decay=1e-2, eps=1e-9)
trainer = Trainer(model, synthdata, optimizer, batch_size=10)
```
-----------------




### Highlighting folder:
This is the "0.1" version of it. It has been my first LLM project ever so it came up very messy and unorganized. I'm now refactoring everything in the xent folder for it to be more flexible and scalable for future development. In the meantime, this folder contains almost everything I did so far. \

To make it work you need to copy/move the base models in the highlighting/models folder 

--------

### MISSING

- Generalize the seed datasets, for now we only have wikipedia taken directly from huggingface datasets 
