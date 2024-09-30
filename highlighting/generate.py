
import pickle
import random
from tqdm import tqdm

from highlight import *
from highlight import XentLang as X

def generate_data():
    
    task_manager = HilightWiki()
    textok, t1, t2 = task_manager.extract()

    min_y_window = int(sub_window_size)
    i = random.randint(0, textok.size(-1) - min_y_window)
    y = textok[0][i:i+min_y_window]

    hilisorted = task_manager.xent_sort(y.unsqueeze(0), t1, t2)

    response = detokenize_l(hilisorted)

    keep_best = 20

    red_best = X().comma.join(response[:keep_best])
    blue_best = X().comma.join(response[-keep_best:])

    min_y_window = int(sub_window_size)

    i = random.randint(0, textok.size(-1) - min_y_window)
    slice = textok[0][i:i+min_y_window]
    ywidth = 3
    o = slice[:ywidth]
    c = slice[-ywidth:]

    y = (detokenize_w(o), detokenize_w(c))
    x1 = detokenize_0(t1)
    x2 = detokenize_0(t2)

    text = detokenize_0(textok)
    func = redblue(y, x1, x2)

    output = text + "\n" + func + "\n" + red_best + X().comma + blue_best

    return output


def generate_dataset(dataset_size: int):
    return [generate_data() for _ in tqdm(range(dataset_size))]


def save_dataset(path: str, dataset: list[str]):
    """ path should be the .pkl file name """
    with open(path, "ab") as f:
        pickle.dump(dataset, f)

