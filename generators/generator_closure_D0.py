
import os
home_dir = os.path.expanduser("~")
work_dir = os.path.join(home_dir, "synth")
from tqdm import tqdm

import torch

from xent.tasks import Closure
from xent.models import M
from xent.dataprocessing import Wikipedia, DataProcessor


task_name = "closure"
n_samples = 1e6
save_truncate = 2e5

model_0 = M("gpt2", "M0", base="base") # load the base modelI
datasource = Wikipedia()
task_0 = Closure(model_0)

new_dataset_name = "closure_0"
dataset_generator = task_0.dataset_generator(datasource.get_random_article_text, out_type="tensor")

save_checkpoint = 1
empty_tensor = torch.zeros((int(save_truncate), model_0.ctx_window)).int() # make sure it's int or you get floats in it for some reason
generated_points = empty_tensor

iter_track = tqdm(total = n_samples//save_truncate + (1 if n_samples%save_truncate != 0 else 0), desc = f"file n°")

# generation loop
for i, sample in enumerate(dataset_generator(n_samples)):
    if i % save_truncate == 0 and i > 0:
        iter_track.update(1)
        save_path = os.path.join(DataProcessor.data_dir, "closure", "D0", f"{new_dataset_name}_{str(save_checkpoint).zfill(2)}.pkl")
        DataProcessor.pickle_dump(generated_points, save_path)
        save_checkpoint += 1
        generated_points = empty_tensor
    generated_points[int(i%save_truncate),:] = sample
    
# save the last iteration if there are leftovers
if torch.equal(generated_points, empty_tensor):
    iter_track.update(1)
    save_path = os.path.join(DataProcessor.data_dir, "closure", "D0", f"{new_dataset_name}_{str(save_checkpoint).zfill(2)}.pkl")
    DataProcessor.pickle_dump(generated_points, save_path)

    