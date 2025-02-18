import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetuned_generation import load_finetuned_llama, generate_with_temp

device_base = "cuda:2"
device_fine = "cuda:2"
tokenizer = AutoTokenizer.from_pretrained("/rcp/marco/models/base/llama-3.2-3B-Instruct")
# model_base = AutoModelForCausalLM.from_pretrained("/rcp/marco/models/base/llama-3.2-3B-Instruct")
# model_base.to(device_base)

system_prompt = """

You are an assistant that provides useful answers to the user.

"""

user_prompt = """Who's the author of Harry potter?"""

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
outputs = generate_with_temp(model_fine, tokens, max_tokens=10000, temperature=0)
jsonstring = tokenizer.decode(outputs[0][tokens.shape[1]+4:-1])
print(jsonstring)
# response_fine = json.loads(jsonstring)
# print(json.dumps(response_fine, indent=2))

