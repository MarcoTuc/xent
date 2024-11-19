import os
from datetime import datetime

import wandb
from tqdm import tqdm 

import torch
from torch.optim import AdamW 

from xent import M, T
from xent.datasets import SynthProcessor
from xent.utils import scalinglaws_lr_function

launched_on = datetime.now().strftime('%d-%b_%H:%M')
message_in_a_bottle = """

Training GPT2-base on 3e6 closure samples

"""

SEED = 21

project_name = "crosstalk"
experiment_name = "base2closure"
wandb_name = f"{project_name}_{experiment_name}_{launched_on}"

# define model to train
base_model = "gpt2"
model_version = "M0"
new_model = "gpt2-base2closure"

# define the training data
base_data = "parallel_parallel"
data_task = "fwd_closure_i"
dataset_size = 3400000 # we use 3.4 million data points in these experiments
train_size = 3000000 # 3 million go to training 
test_size = 400000 # 400k go to testing

# define the training loop
batch_size = 16 #data per training step
train_for = 500 #training steps in between each evaluation
eval_for = 75 #eval steps in between each training loop -- 1200 random samples for each evaluation
sample_every = 1000 #generate a sample every number of training steps
eval_size = eval_for*batch_size #data samples in an eval

# define the optimization
nparams_gpt2 = 124e6
learning_rate = 6.4e-4 #scalinglaws_lr_function(nparams_gpt2)
warmup_steps = 3000

epochs = 1


##########################################################
torch.manual_seed(SEED)

model = M(base_model, model_version)

data  = SynthProcessor(
    base=base_data,
    dataset_name=data_task,
    # files_to_load=5
)
# cut the dataset
data.dataset = data.dataset[:dataset_size]
# shuffle the dataset
data.dataset = data.dataset[torch.randperm(data.dataset.shape[0])]
# train test split
train_set, test_set = data.dataset[:train_size], data.dataset[train_size:train_size+test_size]

optimizer = AdamW(
    model.model.parameters(),
    lr=learning_rate)

