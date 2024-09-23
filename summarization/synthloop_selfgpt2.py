import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from synth import CrossEntropyDifferential


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

prompt = f"""{Y}::TLDR::"""
input = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

def generate_suggestion():
    return model.generate(
        **input,
        max_new_tokens=16,
        temperature=1.2,
        do_sample=True, # uncomment to enable fancy sampling of the output distribution
        no_repeat_ngram_size=2,
        num_return_sequences=200,
        pad_token_id=model.config.eos_token_id
    )

def rank_suggestion(suggestion):
    Ysuggestion = tokenizer.decode(suggestion, skip_special_tokens=True)
    suggestion = Ysuggestion.split("::TLDR::")[1]
    return Ysuggestion, cross_entropy_differential(suggestion, Y, diff=True)

suggestions = generate_suggestion()

n_gen = 40000
gamma = -0.2
new_suggestions = []
barra = tqdm(total=n_gen)

while len(new_suggestions) < n_gen:
    suggestions = generate_suggestion()
    for suggestion in suggestions:
        Ysuggestion, rank = rank_suggestion(suggestion)
        if rank < gamma:
            barra.update(1)
            new_suggestions.append({Ysuggestion:rank.item()})

barra.close()

with open('pizza_selfgpt2_synthdataset.pkl', 'wb') as f:
    pickle.dump(new_suggestions, f)