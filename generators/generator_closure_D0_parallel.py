import os
from multiprocessing import Process
import torch.multiprocessing as mp
from tqdm import tqdm

import torch

from xent.config import *
from xent.tasks import Closure
from xent import M
from xent.datasets import Wikipedia, DataProcessor

task_name = "closure"
data_name = "D0-inverse"
n_samples = 8e6
save_truncate = 8e5
out_type= "tokens"

xent_order_inverse = True

def generate_samples(process_id, n_samples_per_process):
    print(f"Process {process_id} started")
    # Set different seeds for each process
    torch.manual_seed(42 + process_id)
        
    model_0 = M("gpt2", "M0", base="base")
    datasource = Wikipedia()
    task_0 = Closure(model_0)
    
    dataset_generator = task_0.dataset_generator(datasource.get_random_article_text, out_type=out_type, inverse_order=xent_order_inverse)
    
    empty_tensor = torch.zeros((int(save_truncate), model_0.ctx_window)).int()
    generated_points = empty_tensor
    
    base_save_dir = os.path.join(data_dir, task_name, data_name)
    save_checkpoint = 0
    
    # generation loop
    pbar = tqdm(total=n_samples_per_process, 
                desc=f"Process {process_id}",
                position=process_id)  # Position ensures bars don't overlap
                
    for i, sample in enumerate(dataset_generator(n_samples_per_process)):
        if i % save_truncate == 0 and i > 0:
            save_name = f"{data_name}_{str(save_checkpoint).zfill(2)}_proc{process_id}"
            print(f"Process {process_id}: Saving checkpoint {save_checkpoint}")
            DataProcessor.pickle_dump(generated_points, base_save_dir, save_name)
            save_checkpoint += 1
            generated_points = empty_tensor
        generated_points[int(i%save_truncate),:] = sample
        pbar.update(1)
    
    pbar.close()
    print(f"Process {process_id} completed")
    
    # save the last iteration if there are leftovers
    if torch.equal(generated_points, empty_tensor):
        save_name = f"{data_name}_{str(save_checkpoint).zfill(2)}_proc{process_id}"
        DataProcessor.pickle_dump(generated_points, base_save_dir, save_name)

    DataProcessor.save_info_json(
        save_info = {
            "data_type": out_type,
        },
        path = base_save_dir
    )

if __name__ == '__main__':
    n_processes = 3
    n_samples_per_process = int(n_samples) // n_processes  # Split total samples across processes
    
    # Initialize process pool
    mp.set_start_method('spawn')  # Required for CUDA
    processes = []
    
    # Start processes
    for i in range(n_processes):
        p = Process(target=generate_samples, args=(i, n_samples_per_process))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()