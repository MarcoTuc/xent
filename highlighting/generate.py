
import pickle
import random
from tqdm import tqdm

from highlight import *
from highlight import XentLang as X

def generate_data(model, tokenizer):

    HW = HilightWiki(model, tokenizer)
    textok, t1, t2 = HW.extract()

    min_y_window = int(HW.sub_window_size)
    i = random.randint(0, textok.size(-1) - min_y_window)
    y = textok[0][i:i+min_y_window]

    hilisorted = HW.xent_sort(y.unsqueeze(0), t1, t2)

    response = HW.detokenize_l(hilisorted)

    keep_best = 20

    red_best = X().comma.join(response[:keep_best])
    blue_best = X().comma.join(response[-keep_best:])

    min_y_window = int(HW.sub_window_size)

    i = random.randint(0, textok.size(-1) - min_y_window)
    slice = textok[0][i:i+min_y_window]
    ywidth = 3
    o = slice[:ywidth]
    c = slice[-ywidth:]

    y = (HW.detokenize_w(o), HW.detokenize_w(c))
    x1 = HW.detokenize_0(t1)
    x2 = HW.detokenize_0(t2)

    text = HW.detokenize_0(textok)
    func = X().redblue(y, x1, x2)

    output = text + "\n" + func + "\n" + red_best + X().comma + blue_best

    return output


def generate_dataset(model, tokenizer, dataset_size: int):
    return [generate_data(model, tokenizer) for _ in tqdm(range(dataset_size))]


def save_dataset(path: str, dataset: list[str]):
    """ path should be the .pkl file name """
    with open(path, "ab") as f:
        pickle.dump(dataset, f)

