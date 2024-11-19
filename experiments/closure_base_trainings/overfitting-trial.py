message = """ Training gpt2 on a whole 1e6 samples closure-task dataset """

import os
from datetime import datetime

import wandb
from tqdm import tqdm

from torch.optim import AdamW

from xent.models import M
from xent.datasets import SynthProcessor
from xent.trainer import Trainer

project_name = "scaling"
experiment_name = f"{project_name}_{datetime.now().strftime('%d-%b_%H:%M')}"

base_model = "gpt2"
model_version = "M0"
new_model_version = "S1-1000"

task_name = "closure"
data_version = "D0-huge"
cut_dataset = 100
train_split = 0.5
eval_size = 50 # how many points you eval on
batch_size = 10 # depends on available memory

# EVALUATION AND GENERATION INTERVALS
# they refer to batches and not to data samples, so you:
log_interval = 5 # eval every log_interval*batch_size samples from the training set
sample_interval = 5 # same as log but for generation

LEARNING_RATE = 6e-4

EPOCHS = 100

model = M(
    base_model, 
    model_version
    )

synthdata = SynthProcessor(
            task_name, 
            data_version, 
            train_split=train_split, 
            cut_dataset=cut_dataset
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