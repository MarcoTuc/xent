message = """ Training GPT2 on 800 D0-correct samples to check for dataset-size scaling laws"""

import os
from datetime import datetime

import wandb
from tqdm import tqdm

from torch.optim import AdamW

from xent import M
from xent.datasets import SynthProcessor
from xent.trainer import Trainer

project_name = "scaling"
scaling_name = "9e6"
experiment_name = f"{project_name}_{scaling_name}_{datetime.now().strftime('%d-%b_%H:%M')}"

base_model = "gpt2"
model_version = "M0"

task_name = "closure"
data_version = "D0-correct-big"
cut_trainset = int(float(scaling_name))
eval_size = 1000 # how many points you eval on
batch_size = 10 # depends on available memory

new_model_version = f"S1-{scaling_name}"

# EVALUATION AND GENERATION INTERVALS
# they refer to batches and not to data samples, so you:
log_interval = 80 # eval every log_interval*batch_size samples from the training set
sample_interval = 40 # same as log but for generation

LEARNING_RATE = 6e-4

FIXED_COMPUTE_STEPS = 9e5

EPOCHS = int(FIXED_COMPUTE_STEPS * batch_size / cut_trainset)

print(f"Training for epochs: {EPOCHS}")

model = M(
    base_model, 
    model_version
    )

synthdata = SynthProcessor(
            task_name, 
            data_version, 
            split_posit=9000000, # dataset D0-correct has 9600000 samples, of which we use 9000000 for training and 600000 for testing
            cut_trainset=cut_trainset
            )

optimizer = AdamW(
            model.model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=1e-1, 
            eps=1e-9
            )

trainer = Trainer(
            model, 
            synthdata, 
            optimizer, 
            batch_size=batch_size, 
            eval_size=eval_size,
            log_interval=log_interval,
            make_samples=True,
            sample_interval=sample_interval
            )

saving_options = {
    "base": task_name, # the base will tell inside which folder to put the new model
    "new_version": new_model_version 
}

wandb.init(
    project = project_name,
    name = experiment_name,
    config={
        "model": base_model,
        "model_version": model_version,
        "task": task_name,
        "data": data_version,
        "epochs": EPOCHS
    }
)

for epoch in range(EPOCHS):
    wandb.log({"epoch": epoch})
    trainer.train_with_validation(saving_options=saving_options, tot_epochs=EPOCHS)
    trainer.update_epoch(1)

wandb.finish()