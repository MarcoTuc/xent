import os 
work_dir = os.path.join(os.getcwd(), "highlighting")
models_path = os.path.join(work_dir, "models")
data_path = os.path.join(work_dir, "data")
from generate import generate_dataset, save_dataset
from evolve import load_model_and_tokenizer

model_path = os.path.join(models_path, "gpt2-xl-M0")
M0, tokenizer = load_model_and_tokenizer(model_path)

D0 = generate_dataset(M0, tokenizer, 10, keep_best=25)
datasave_path = os.path.join(data_path, "D0.pkl")
save_dataset(datasave_path, D0)
