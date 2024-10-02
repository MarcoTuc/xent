import pickle
import random
from tqdm import tqdm

from highlight import *
from highlight import XentLang as X

X = X()

def generate_data(model, tokenizer, keep_best):

    HW = HilightWiki(model, tokenizer)
    textok, t1, t2 = HW.extract()

    min_y_window = int(HW.sub_window_size)
    i = random.randint(0, textok.size(-1) - min_y_window)
    y = textok[0][i:i+min_y_window]

    if random.random() < 0.5: 
        t1, t2 = t2, t1

    hilisorted = HW.xent_sort(y.unsqueeze(0), t1, t2)
    response = HW.detokenize_l(hilisorted)

    ywidth = 3
    open_y = y[:ywidth]
    close_y = y[-ywidth:]

    ys = (HW.detokenize_w(open_y), HW.detokenize_w(close_y))
    x1 = HW.detokenize_0(t1)
    x2 = HW.detokenize_0(t2)

    text = HW.detokenize_0(textok)
    func = X.redblue(ys, x1, x2)

    if keep_best:
        red_best = X.comma.join(response[:keep_best])
        blue_best = X.comma.join(response[-keep_best:])
        output = text + "\n" + func + "\n" + red_best + X.comma + blue_best
    else:
        output = text + "\n" + func + "\n" + X.comma.join(response)

    return output


def generate_dataset(model, tokenizer, dataset_size: int, keep_best=20):
    return [generate_data(model, tokenizer, keep_best) for _ in tqdm(range(dataset_size))]


def save_dataset(path: str, dataset: list[str]):
    """ path should be the .pkl file name """
    with open(path, "ab") as f:
        pickle.dump(dataset, f)


def load_dataset(path: str):
    with open(path, "rb") as data:
        return pickle.load(data)

