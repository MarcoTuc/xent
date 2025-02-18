import torch
import torch.nn.functional as F 
from torchtune.models.llama3_2 import llama3_2_3b
from torchtune.training import FullModelHFCheckpointer
from transformers import AutoTokenizer

def load_finetuned_llama(device):
    checkpoint_dir = output_dir = "/rcp/marco/models/llama-3.2-3B-Instruct/closure-finetuned/checkpoints"
    pytorch_files = [
    "hf_model_0001_0.pt",
    "hf_model_0002_0.pt",
    ]

    # Set up the checkpointer and load state dict
    checkpointer = FullModelHFCheckpointer(
    checkpoint_dir=checkpoint_dir,
    checkpoint_files=pytorch_files,
    output_dir=output_dir,
    model_type="LLAMA3_2",
    )

    torchtune_sd = checkpointer.load_checkpoint()
    model = llama3_2_3b()
    model.load_state_dict(torchtune_sd["model"])
    model.to(device)
    return model

tokenizer = AutoTokenizer.from_pretrained("/rcp/marco/models/base/llama-3.2-3B-Instruct")

def generate_with_temp(
        model, 
        prompt, 
        max_tokens, 
        device=None,
        temperature=0.7, 
        skip_special_tokens=False,
        return_text=False,
): 
    if isinstance(prompt, str): tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    elif isinstance(prompt, torch.Tensor): tokens = prompt
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :]
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    if skip_special_tokens:
        # Create a mask for non-special tokens
        mask = torch.tensor([token not in tokenizer.all_special_ids for token in tokens[0]], device=device)
        # Filter out special tokens
        tokens = tokens[:, mask]
    
    if return_text: 
        return tokenizer.decode(tokens[0])

    return tokens