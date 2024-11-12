from tqdm import trange, tqdm
from time import sleep

for i in tqdm(range(4), desc='1st loop'):
    for j in tqdm(range(5), desc='2nd loop', leave=True):
        for k in tqdm(range(50), desc='3rd loop', leave=False):
            sleep(0.01)