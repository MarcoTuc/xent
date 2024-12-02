import os
from tqdm import tqdm
import torch

import datasets
from datasets import load_dataset

from xent import M
from xent.datasets import DataProcessor
from xent.config import *

model = M("base", "gpt2-xl", "M0")

data = load_dataset(
    "wikipedia", 
    "20220301.en", 
    trust_remote_code=True,
    )["train"]

# this is needed to reproduce how I separated the data in the parallel_parallel
# generator, since I want to pre-train the model on that corpus I'm splitting it 
# in the same way so we have a reproducible thing. 
# the "shards" parameter is needed to define to which parallel_parallel checkpoint 
# one wants to reproduce the corpus dataset. 
# For now I aim to pre-train on 90 checkpoints. 
n_shards = 600
shards = 90
data_1 = data.shard(num_shards=3, index=0)
data_2 = data.shard(num_shards=3, index=1)
data_3 = data.shard(num_shards=3, index=2)
finaldata = []
for shard in range(shards):
    finaldata.append(data_1.shard(num_shards=n_shards, index=shard))
    finaldata.append(data_2.shard(num_shards=n_shards, index=shard))
    finaldata.append(data_3.shard(num_shards=n_shards, index=shard))
finaldata = datasets.concatenate_datasets(finaldata)
finaldata = finaldata.train_test_split(test_size=0.1, shuffle=False)


# here define how the new dataset has to be saved 
shards_to_save = 16
base_save_dir = "wiki90-tok"

# tokenize and save the training set
splits = ["train", "test"]
for split in splits:
    for sh in tqdm(range(shards_to_save)):
        tensor_to_save = torch.LongTensor([]).to(device)
        dshard = finaldata[split].shard(num_shards=shards_to_save, index=sh)
        for el in tqdm(dshard, leave=False):
            tensor_to_save = torch.cat([
                tensor_to_save, 
                model.tokenize(el["text"]).input_ids, 
                torch.LongTensor([model.tokenizer.eos_token_id]).unsqueeze(0).to(device)
                ], dim=-1)  
        save_dir = os.path.join(base_save_dir, split)
        save_name = f"{split}_{str(sh).zfill(4)}"
        DataProcessor.pickle_dump(
            tensor_to_save,
            save_dir,
            save_name
        )