### SETUP

1. create an environment inside this folder and call it menv: `python -m venv -n menv`

2. activate the environment `source menv/bin/activate`

3. Install the xent package inside the menv environent: `python setup.py develop`

4. Populate the models folder with the base models you can use *the model_downloader.ipynb* notebook

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

### Highlighting folder:
This is the "0.1" version of it. It has been my first LLM project ever so it came up very messy and unorganized. I'm now refactoring everything in the xent folder for it to be more flexible and scalable for future development. In the meantime, this folder contains almost everything I did so far. \

To make it work you need to copy/move the base models in the highlighting/models folder 

--------

### MISSING

- Generalize the seed datasets, for now we only have wikipedia taken directly from huggingface datasets 
