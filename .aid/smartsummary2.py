import os
import torch, torch.nn as nn, torch.nn.functional as F

from tqdm import tqdm

import llama_cpp
import random    


home_dir = os.path.expanduser('~')
gguf_dir = os.path.join(home_dir, 'gguf')

model_file_names = dict(gpt_q8="gpt2.Q8_0.gguf", gemma_q8="gemma-2-2b-Q8_0.gguf", nemo_f16="NemoMix-Unleashed-12B-f16.gguf", phi_q8="Phi-3.5-mini-Instruct-Q8_0.gguf")


def load_models_by_name(model_file_names, *, gguf_dir):
    model_paths = { name:os.path.join(gguf_dir, model_file_names[name]) for name in model_file_names }
    llama_model_settings = dict(n_gpu_layers=-1, verbose=False, logits_all=True)
    models_by_name = { name: llama_cpp.Llama(model_path=model_paths[name], **llama_model_settings) for name in model_paths }
    return models_by_name

models_by_name = load_models_by_name(model_file_names, gguf_dir=gguf_dir)

model = models_by_name['gemma_q8']
i_model = models_by_name['phi_q8']

def tokenize(model, string): return torch.tensor(model.tokenize(string.encode('utf-8'))) # [t]
def detokenize(model, tokens): return model.detokenize(tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens)

def to_list(tokens): return tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
def to_tensor(tokens): return tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.int64)


def comp_logits(model, tokens):
    if isinstance(tokens, torch.Tensor): tokens = tokens.tolist()
    model.reset()
    model.eval(tokens)
    return torch.tensor(model.eval_logits)

def compute_xent(model, tokens, prefix_length=0): 
    tokens = to_tensor(tokens)
    logits = comp_logits(model, tokens)
    xent = float(F.cross_entropy(logits[prefix_length:-1,:], tokens[prefix_length + 1:]))
    return xent

def compute_xent_with_prefix(model, tokens, prefix_tokens):
    return compute_xent(model, to_list(prefix_tokens) + to_list(tokens), len(to_list(prefix_tokens)))

string="Hello, it's a world of wonders, and a universe of magic, and I wish you a warm welcome in it. There are plenty of exciting things to do in this place! Please help yourself, and now let's begin our adventures!"


string = """Then ensued a slight anticlimax, for the Head, though a very strong man
in hand and arm, found it impossible to do as he had designed, and tear
three hundred pages across and across. But the form generally, and Mr.
Dutton in particular, were too stiff with horror to notice it. So,
instead, the Head tore off section after section of the lightly sewn
leaves, instead of tearing the pages, laid the dismembered carrion on
the floor, and stamped on it, and then, with an indescribable
ejaculation of disgust, threw the mutilated remains into the fireplace."""


tokens = tokenize(model, string)

def quick_xent(string, prefix):
    string_tokens = tokenize(model, string) if isinstance(string, str) else string
    prefix_tokens = tokenize(model, prefix) if isinstance(prefix, str) else prefix
    return compute_xent_with_prefix(model, string_tokens, prefix_tokens) 

def quick_xent_on_suggestions(string, suggestions):
    basic_xent = quick_xent(string, "")
    for suggestion in suggestions:
        diff_xent = basic_xent - quick_xent(string, suggestion)

poss_tokens = list(set(to_list(tokenize(model, string))))

def remove_chars(string, chars):
    for c in chars: string = string.replace(c, '')
    return string

def make_i_suggestions(imodel, string, n_sugg=10000, n_words=4): # use an instruct model
    query = f"Can you summarize the following in {n_words} words (no punctuation): {string}"
    suggestions = []
    for i in tqdm(range(n_sugg)):
        raw_suggestion = imodel(query, temperature=1.2)['choices'][0]['text']
        raw_suggestion = raw_suggestion.replace('\n', ' ')
        raw_suggestion = remove_chars(raw_suggestion, '.,:;#\'\"!?`-*')
        raw_words = [w.lower() for w in raw_suggestion.split(' ') if len(w) > 1]
        suggestion = ' '.join(raw_words[max(len(raw_words) - n_words, 0) - 1: - 1])
        suggestions.append(suggestion)
    return suggestions


def make_random_suggestions(poss_tokens, n_sugg=10000, l_sugg=4):
    for i in range(n_sugg):
        suggestion = []
        for k in range(l_sugg): 
            suggestion.append(random.choice(poss_tokens))
    suggestions.append(suggestion)
    return suggestions


def quick_opt_prefix(tokens, suggestions):
    basic_xent = quick_xent(tokens, "")
    print(f"{basic_xent=}")
    max_diff = 0.0
    sum_diffs = 0.0

    for (i, suggestion) in enumerate(suggestions):
        diff_xent = basic_xent - quick_xent(string, suggestion)
        sum_diffs += diff_xent
        if diff_xent > max_diff:
            max_diff = diff_xent
            print(i, ":", f"{suggestion}", ":", max_diff, ":", sum_diffs / (i + 1))


n_sugg = 1000
n_words = 5

suggestions = make_i_suggestions(i_model, string, n_sugg, n_words)

for suggestion in suggestions: print(suggestion)

quick_opt_prefix(string, suggestions)


