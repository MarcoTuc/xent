import os
import sys
from datetime import datetime

import wandb
from tqdm import tqdm 

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from xent import M, T
from xent.datasets import SynthProcessor
from xent.utils import scalinglaws_lr_function, Tee

launched_on = datetime.now().strftime('%d-%b_%H:%M')
message_in_a_bottle = """

Training GPT2-base on 3e6 closure samples

"""

use_wandb = True

SEED = 21

project_name = "crosstalk"
experiment_name = "base2closure"
wandb_name = f"{project_name}_{experiment_name}_{launched_on}"

# define model to train
base_model = "gpt2"
model_version = "M0"
new_model_base = "crosstalk-cluster"
new_model_name = "gpt2-base2closure"

# define the training data
base_data = "parallel_parallel"
data_task = "fwd_closure_i"
dataset_size = 3400000 # we use 3.4 million data points in these experiments
train_size = 3000000 # 3 million go to training 
test_size = 400000 # 400k go to testing

# define the training loop
batch_size = 40 #data per training step
train_for = 100 #training steps in between each evaluation
eval_for = 30 #eval steps in between each training loop -- 1200 random samples for each evaluation
sample_every = 500 #generate a sample every number of training steps
eval_size = eval_for*batch_size #data samples in an eval

# define the optimization
nparams_gpt2 = 124e6
learning_rate = 6.3e-4 #scalinglaws_lr_function(nparams_gpt2)
warmup_steps = 3000 #very simple linear warmup

epochs = 1


##########################################################
stdout_dir = os.path.join(os.getenv("XENT_MODELS_PATH"), new_model_base, base_model, new_model_name, "console.txt")
os.makedirs(os.path.dirname(stdout_dir), exist_ok=True)
f = open(stdout_dir, "w+")
sys.stdout = Tee(f) 
sys.stderr = Tee(f)

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
    make_samples=True,
    sample_interval=sample_every,
    report_wandb=use_wandb
)

saving_options = {
    "base": new_model_base,
    "model_name": base_model,
    "new_version": new_model_name
}

saving_info = {
        "model": base_model,
        "model_version": model_version,
        "synth_base": base_data,
        "data_task": data_task,
        "data_samples": dataset_size,
        "train_samples": train_size,
        "test_samples": test_size,
        "message": message_in_a_bottle
    }

if use_wandb:
    wandb.init(
        project=project_name,
        name=wandb_name,
        config=saving_info
)

for epoch in range(epochs):
    trainer.train_with_validation(
        saving_options=saving_options,
        saving_info=saving_info,
        tot_epochs=epochs
    )
    trainer.update_epoch(1)

if use_wandb: wandb.finish()