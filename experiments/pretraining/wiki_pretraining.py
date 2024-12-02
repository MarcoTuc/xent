import os
import sys
from datetime import datetime

import wandb
from tqdm import tqdm 

import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.amp import autocast, GradScaler

import datasets
from datasets import load_dataset

from xent import (
    Model as M,
    Trainer as T
)

from xent.datasets import SynthProcessor, DataProcessor
from xent.utils import Tee
from xent.config import *

launched_on = datetime.now().strftime('%d-%b_%H:%M')
message_in_a_bottle = """

Pretraining of GPT2-XL on Wikipedia

"""

use_wandb = True
log_intxt = False

SEED = 21

project_name = "pretraining"
experiment_name = "wikipedia"
wandb_name = f"{experiment_name}_{launched_on}"

# define model to train
model_base = "base"
model_name = "gpt2-xl"
model_version = "M0"
new_model_base = "pretrained"
new_model_version = "gpt2-xl-wikipedia-local"

# define the training loop
batch_size = 4 #data per training step
train_for = 10 #training steps in between each evaluation
eval_for = 5 #eval steps in between each training loop -- 1200 random samples for each evaluation
sample_every = 500 #generate a sample every number of training steps
eval_size = eval_for*batch_size #data samples in an eval

# define the optimization
learning_rate = 6e-4
warmup_steps = 5000 #very simple linear warmup

training_steps = 100000

##########################################################
if log_intxt:
    stdout_path = os.path.join(os.getenv("XENT_MODELS_PATH"), new_model_base, model_name, new_model_version, "console.txt")
    tqdm_path = os.path.join(os.getenv("XENT_MODELS_PATH"), new_model_base, model_name, new_model_version, "tqdm.txt")
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
    os.makedirs(os.path.dirname(tqdm_path), exist_ok=True)
    f = open(stdout_path, "w+")
    t = open(tqdm_path, "w+")
    sys.stdout = Tee(f) 
    sys.stderr = Tee(t)

torch.manual_seed(SEED)

model = M(model_base, model_name, model_version)

train_set = SynthProcessor(
    base="wiki90-tok",
    dataset_name="train"
).dataset

test_set = SynthProcessor(
    base="wiki90-tok",
    dataset_name="test"
).dataset

train_set_cpu = torch.cat(train_set, dim=-1).to("cpu")
test_set_cpu = torch.cat(test_set, dim=-1).to("cpu")
del train_set; del test_set; torch.cuda.empty_cache()

class Wikichunk(Dataset):
    def __init__(self, tensor, ctx=1024):
        self.data = tensor
        self.ctx = ctx
        self.length = tensor.shape[0] - ctx + 1
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = torch.randint(0, self.length, (1,)).item()
        chunk = self.data[idx:idx+self.ctx]
        return chunk

train_set = Wikichunk(train_set_cpu, ctx=1024)
test_set = Wikichunk(test_set_cpu, ctx=1024)

optimizer = AdamW(
    model.model.parameters(),
    lr=learning_rate)

scheduler = LinearLR(
    optimizer,
    start_factor=1e-9,
    end_factor=1,
    total_iters=warmup_steps
)

trainer = T(
    initial_model=model,
    train_set=train_set,
    test_set=test_set,
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=batch_size,
    eval_size=eval_size,
    log_interval=train_for,
    make_samples=False,
    sample_interval=sample_every,
    report_wandb=use_wandb
)

saving_options = {
    "base": new_model_base,
    "model_name": model_name,
    "new_version": new_model_version
}

saving_info = {
        "model": model_base,
        "model_version": model_version,
        "new_model": new_model_version,
        "trained_for": training_steps,
        "message": message_in_a_bottle
    }

if use_wandb:
    wandb.init(
        project=project_name,
        name=wandb_name,
        config=saving_info
    )

scaler = GradScaler()

iter = 0
with autocast("cuda"):
    while True: 
        trainer.pre_train(
            saving_info=saving_info,
            saving_options=saving_options,
            steps=training_steps
        )
        break

if use_wandb: 
    wandb.finish()