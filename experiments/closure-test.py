message = """ Training gpt2 on a whole 1e6 samples closure-task dataset """

import os
from datetime import datetime

import wandb
from tqdm import tqdm

from torch.optim import AdamW

from xent.models import M
from xent.dataprocessing import SynthProcessor
from xent.trainer import Trainer

now = datetime.now()
project_name = "closure-big"
experiment_name = f"{project_name}_{now.strftime('%d-%b_%H:%M')}"

base_model = "gpt2"
model_version = "M0"
new_model_version = "M1-big-test"

task_name = "closure"
data_version = "D0"
cut_dataset = 200
train_split = 0.5

EPOCHS = 8
batch_size = 10
log_interval = 2 # also works as evaluation interval
sample_interval = 2 # also works as evaluation interval

LEARNING_RATE = 6e-4
beta1 = 0.9
beta2 = 0.95
decay = 1e-1


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
            betas=(beta1, beta2), 
            weight_decay=decay, 
            eps=1e-9
            )

trainer = Trainer(
            model, 
            synthdata, 
            optimizer, 
            batch_size=batch_size, 
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

for epoch in tqdm(range(EPOCHS)):
    wandb.log({"epoch": epoch})
    trainer.train_with_validation(saving_options=saving_options)
    trainer.update_epoch(1)

wandb.finish()