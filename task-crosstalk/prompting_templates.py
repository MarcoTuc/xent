import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetuned_generation import load_finetuned_llama, generate_with_temp

device_base = "cuda:2"
device_fine = "cuda:2"
tokenizer = AutoTokenizer.from_pretrained("/rcp/marco/models/base/llama-3.2-3B-Instruct")
# model_base = AutoModelForCausalLM.from_pretrained("/rcp/marco/models/base/llama-3.2-3B-Instruct")
# model_base.to(device_base)

system_prompt = """You are going to count the number of words in the user prompt and output them in an ordered JSON format like in the following example. You are very precise and start counting from the first word in the user prompt until the last one. Make sure to include all the words in the user text, especially don't forget the words at the beginning and at the end. DO NOT output anything else than the JSON. DO NOT change the phrase of the user. JUST OUTPUT THE JSON. 

EXAMPLE BELOW: 

User: Hey what's up? My name is John, nice to meet you! 

Response:
{
    "words" : [
        {"word": "Hey", "position": 1},
        {"word": "what's", "position": 2},
        {"word": "up?", "position": 3},
        {"word": "My", "position": 4},
        {"word": "name", "position": 5},
        {"word": "is", "position": 6},
        {"word": "John", "position": 7},
        {"word": "nice", "position": 8},
        {"word": "to", "position": 9},
        {"word": "meet", "position": 10},
        {"word": "you!", "position": 11},
        {"word": "Hey", "position": 12},
        {"word": "again!", "position": 13}
    ]
}

END OF EXAMPLE. 

Make sure to exactly follow this provided structure and do not follow any other structure you may have been shown during any finetuning training. 




"""

user_prompt = """Since the release of the first novel, Harry Potter and the Philosopher's Stone, on 26 June 1997, the books have found immense popularity and commercial success worldwide. They have attracted a wide adult audience as well as younger readers and are widely considered cornerstones of modern literature."""

tokens = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{user_prompt}"}
            ],
            tokenize=True,
            return_tensors="pt"
        ).to(device_base)

# print("\n\n Vanilla response ----------------------------------------------------")
# outputs = model_base.generate(tokens, max_new_tokens=10000)
# jsonstring = tokenizer.decode(outputs[0][tokens.shape[1]+4:-1])
# response_base = json.loads(jsonstring)
# print(json.dumps(response_base, indent=2))

# print("\n\n")
# print("deleting base model from memory")
# model_base.to("cpu")
# del model_base
# torch.cuda.empty_cache()
# print("loading finetuned model to memory")
model_fine = load_finetuned_llama(device_fine)
print("finetuned model loaded")

print("\n\n Finetuned response ----------------------------------------------------")
outputs = generate_with_temp(model_fine, tokens, max_tokens=10000, temperature=0.7)
jsonstring = tokenizer.decode(outputs[0][tokens.shape[1]+4:-1])
print(jsonstring)
# response_fine = json.loads(jsonstring)
# print(json.dumps(response_fine, indent=2))

