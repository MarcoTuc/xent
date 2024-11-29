message = """ 

Take the whole wikipedia corpus and use it to parallel production of three tasks: 
1. Forward closure
2. Backward closure
3. Just the xents
4. Just the xents backwards

For each of these tasks, we'll generate two precisions: 
1. Integer rounded
2. 1e-2 precision float

 """

import os
import torch.multiprocessing as mp
from tqdm import tqdm

import torch
from datasets import load_dataset

from xent.config import *
from xent.tasks import Closure
from xent import M
from xent.datasets import DataProcessor

n_processes = 3
n_shards_per_process = 600
shard_checkpoint = 192

data = load_dataset(
    "wikipedia", 
    "20220301.en", 
    trust_remote_code=True,
    )["train"]

base_dir = "parallel_parallel"
base_save_dir = os.path.join(data_dir, base_dir)
out_type= "tokens"

model = M("gpt2", "M0", base="base")

task = Closure(model)

def generate_samples(process_id):
    torch.manual_seed(42 + process_id)
    process_shard = data.shard(num_shards=n_processes, index=process_id)
    for shard in tqdm(range(n_shards_per_process), desc=f"process {process_id} shard", position=process_id*2, leave=False):
        data_shard = process_shard.shard(num_shards=n_shards_per_process, index=shard)
        generated = {
            "fwd_closure_i": torch.LongTensor([]).to(device),
            "fwd_closure_f": torch.LongTensor([]).to(device),
            "xonly_fwd_i": torch.LongTensor([]).to(device),
            "xonly_fwd_f": torch.LongTensor([]).to(device),
            "bwd_closure_i": torch.LongTensor([]).to(device),
            "bwd_closure_f": torch.LongTensor([]).to(device),
            "xonly_bwd_i": torch.LongTensor([]).to(device),
            "xonly_bwd_f": torch.LongTensor([]).to(device),
            "xent_rank_top": torch.LongTensor([]).to(device)
        }
        for d in tqdm(data_shard["text"], desc=f"process {process_id}  data", position=process_id*2+1, leave=False):
            output = task.generate_parallel_parallel(lambda: d)
            generated = {
                key: torch.cat([generated[key], output[key]])
                for key in generated.keys()
            } # keys need to correspond, need to abstract this behavior eventually


        for key in generated.keys():
            save_dir = os.path.join(base_save_dir, key)
            save_name = f"{key}_{str(shard).zfill(4)}_proc{process_id}"
            DataProcessor.pickle_dump(
                generated[key],
                save_dir,
                save_name
            )
            DataProcessor.save_info_json(
                save_info = {
                    "data_type": out_type,
                    "saved_shards": shard,
                    "task_type": key,
                    "example task": model.detokenize(generated[key][0])
                },
                path = save_dir
            )


if __name__ == "__main__":
    
    mp.set_start_method("spawn", force=True)
    processes = []

    for i in range(n_processes):
        p = mp.Process(target=generate_samples, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
