import torch
import torch.nn.functional as F
import llama_cpp
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle
from synth import CrossEntropyDifferential

# instantiating phi to generate suggestions
phi = llama_cpp.Llama(
      model_path="models/Phi-3.5-mini-instruct.Q8_0.gguf",
      n_gpu_layers=-1, 
      verbose=False,
      logits_all=True,
)

# gpt2 to evaluate cross-entropy and train
device = torch.device("cuda")
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
cross_entropy_differential = CrossEntropyDifferential(model, tokenizer, device)


Y = """Pizza is the best thing in the world, all the mozzarella and the tomato sauce on my margherita are amazing.
My Italian friend Angelo told me that his favourite pizza is the quattroformaggi rossa with salsiccia.
When I was a kid, I used to watch the pizza maker create his pizzas, he was from Romania but a very nice gentleman I have to say."""
S = "Write a TL;DR: of the following text in exactly 4 words. No more, no less. Separate words with spaces. End with a . after writing it."
prompt = f"""<|system|>{S}<|end>
<|user|>{Y}<|end|>
<|assistant|>"""


def generate_suggestion():
    return phi(prompt, temperature=4, stop=["."], max_tokens=30)["choices"][0]["text"]

def rank_suggestion(suggestion):
    return cross_entropy_differential(suggestion, Y, diff=True)


n_gen = 5000
gamma = -0.4
new_suggestions = []

barra = tqdm(total=n_gen)

while len(new_suggestions) < n_gen:    
    suggestion = generate_suggestion()
    rank = rank_suggestion(suggestion)
    if rank < gamma:
        barra.update(1)
        new_suggestions.append({f"{Y}::TLDR::{suggestion}":rank.item()})

barra.close()
phi.close()

with open('pizza_synthdataset.pkl', 'wb') as f:
    pickle.dump(new_suggestions, f)

