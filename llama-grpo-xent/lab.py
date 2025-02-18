
import re
import os
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer


PatchFastRL("GRPO", FastLanguageModel)


max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


SYSTEM_PROMPT = """

You estimate the integer-rounded cross-entropy of the text before a special function. 

Given a certain text before the function like:

token_1token_2token_3token_4

You are going to format it like:

token_1: cross_entropy of token_1
token_2: cross_entropy of token_2
token_3: cross_entropy of token_3
token_4: cross_entropy of token_4

Tokens have to follow the same pattern of your own tokenizer.

"""



def format_response(data: str) -> list[dict]:
    """ This function formats a response from xent format to a dictionary of tokens and xents as ints """
    linecond = r"(?<=\d)\n"
    txcond = r":(?=\s\d)"
    r_split = re.split(linecond, data[1:-1])
    tx_split = [re.split(txcond, line) for line in r_split]
    return [
        {
            "token": x[0],
            "xent": int(x[1].strip())
        }   for x in tx_split
    ]

def get_xent_dataset(test_size = 0.1) -> Dataset:
    data_path = "/rcp/marco/data/instruct_llama/closure_database.json"
    data = load_dataset("/rcp/marco/data/instruct_llama", data_files=[data_path], split="train")
    data = data.train_test_split(test_size=test_size, shuffle=True)

    dataset = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["prompt"]}
            ],
            "answer": x["response"]
        },
        remove_columns=["response"]
    )

    return dataset

dataset = get_xent_dataset()


def dummy_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    print(f"{len(responses)} completions have been produced")
    for response in responses: 
        print(response)
        print("\n\n")
    return 0

def formatting_reward(completions, answer, **kwargs):
    """ This reward function makes sure the output has a certain format """
    wf = 1
    regex_line = r"[\S\s]*?:\s\d{1,2}\n"
    # regex_all = r"({regex_line})+" # if you wanna make a whole-thing reward instead of line per line
    responses = [completion[0]["content"] for completion in completions]
    correct = [len(re.findall(regex_line, response)) for response in responses]
    print("-"*40)
    for i, corr in enumerate(correct):
        print(f"Correctly formatted lines on response {i}: {corr}")
        if corr > 0:
            print(responses[i])
    return [c*wf if c > 0 else -5 for c in correct]



training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 3e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 6,
    gradient_accumulation_steps = 1,
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 2048,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 250,
    # save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "trained_models/llama"
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        dummy_reward
    ],
    args = training_args,
    train_dataset = dataset["train"],
)

trainer.train()


