message = """ Training gpt2 on tasks it generates on the go """

import os
from datetime import datetime

import wandb
from tqdm import tqdm

import torch 
from torch.optim import AdamW

from xent.config import *
from xent.models import M
from xent.dataprocessing import Wikipedia
from xent.tasks import Closure
from xent.trainer import Trainer, Evolver

project_name = "closure-longtraining"
experiment_name = f"closure-evolution_{datetime.now().strftime('%d-%b_%H:%M')}"

base_model = "gpt2"
model_version = "M0"
new_model_version = "M1-evolved"
reinitialize = False

task_name = "closure"
batch_size = 4 # depends on available memory, on our V100s we can get to 10
train_split = 0.8
generation_split = 0.5

# EVALUATION AND GENERATION INTERVALS
# they refer to batches and not to data samples, so you:
train_log_interval = valid_log_interval = 5 # eval every log_interval*batch_size samples from the training set
sample_interval = 500 # sample interval

LEARNING_RATE = 6e-4
beta1 = 0.9
beta2 = 0.95
decay = 1e-1

model = M(
    base_model, 
    model_version,
    reinitialize=reinitialize
    )

task = Closure(
    model
)

corpus_generator = Wikipedia(split=train_split)
get_train_sample = corpus_generator.get_random_train_text
get_test_sample = corpus_generator.get_random_test_text

make_train = task.dataset_synthesizer(get_sample=get_train_sample, n_samples=int(2*batch_size/generation_split), out_type="tensor")
make_test = task.dataset_synthesizer(get_sample=get_test_sample, n_samples=int(2*batch_size/generation_split), out_type="tensor")
make_gen = task.dataset_synthesizer(get_sample=get_test_sample, n_samples=1, out_type="tensor")

optimizer = AdamW(
            model.model.parameters(), 
            lr=LEARNING_RATE, 
            betas=(beta1, beta2), 
            weight_decay=decay, 
            eps=1e-9
            )

saving_options = {
    "base": task_name, # the base will tell inside which folder to put the new model
    "new_version": new_model_version 
}

evolver = Evolver(
    model, 
    optimizer
)

wandb.init(
    project = project_name,
    name = experiment_name,
    config={
        "model": base_model,
        "model_version": model_version,
        "task": task_name,
    }
)

iter_counter = 0
iter_bar = tqdm()
empty_loss = torch.tensor([]).to(device)

train_loss = empty_loss
valid_loss = empty_loss
best_valid_loss = float("inf")
gener_loss = empty_loss
gen_table = []

while True:
    iter_counter += 1
    iter_bar.update(1)
    t_loss = evolver.train_batch(make_train())
    train_loss = torch.cat([train_loss, t_loss])
    v_loss = evolver.eval_batch(make_test())
    valid_loss = torch.cat([valid_loss, v_loss])
    gener_loss = torch.cat([gener_loss, v_loss])

    # training reporting
    if iter_counter % train_log_interval == 0:
        wandb.log({"train_loss": train_loss.mean().item()})
        train_loss = empty_loss

    # evaluation reporting
    if iter_counter % valid_log_interval == 0:
        wandb.log({"valid_loss": train_loss.mean().item()})
        valid_loss = empty_loss
        evolver.save_model(task_name, base_model, new_model_version)

    # generation reporting
    if iter_counter % sample_interval == 0:
        prompt, gen_sample = evolver.perform_task(make_gen(), split=True)
        gen_table.append([gener_loss.mean().item(), prompt, gen_sample])
        gener_loss = empty_loss
        wandb.log({"generated_samples": wandb.Table(
                            columns=["loss", "prompt", "output"],
                            data=gen_table,
                            allow_mixed_types=True
                        )})


wandb.finish()