message = """ Training gpt2 on a whole 1e6 samples closure-task dataset """

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

task = "closure"
data_version = "D0"
cut_dataset = 200
train_split = 0.9

EPOCHS = 8

LEARNING_RATE = 6e-4
beta1 = 0.9
beta2 = 0.95
decay = 1e-1

wandb.init(
    project = project_name,
    name = experiment_name,
    config={
        "model": base_model,
        "model_version": model_version,
        "task": task,
        "data": data_version,
        "epochs": EPOCHS
    }
)

model = M(base_model, model_version)
synthdata = SynthProcessor(task, data_version, train_split=train_split, cut_dataset=cut_dataset)
optimizer = AdamW(model.model.parameters(), lr=LEARNING_RATE, betas=(beta1, beta2), weight_decay=decay, eps=1e-9)
trainer = Trainer(
            model, 
            synthdata, 
            optimizer, 
            batch_size=5, 
            log_interval=10,
            make_samples=True,
            sample_interval=5)

for epoch in tqdm(range(EPOCHS)):
    wandb.log({"epoch": epoch})
    trainer.train_with_validation()

wandb.finish()  