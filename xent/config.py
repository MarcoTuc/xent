import os
work_dir = os.getcwd()
models_dir = os.getenv("XENT_MODELS_PATH")
data_dir = os.getenv("XENT_DATA_PATH")

device = "cuda:0"