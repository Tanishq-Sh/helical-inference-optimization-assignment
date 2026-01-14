import os
import torch

RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output") # path for storing perturbed embeddings
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # for dynamic switching between CPU and GPU based on availability 

MODEL_CONFIG = {
    "model_name": "hyenadna-tiny-1k-seqlen-d256", # Model name to use
    # "model_name": "hyenadna-tiny-1k-seqlen", # Alternate Model name to use
    "batch_size": 32, # Number of sequences to process in a batch
    "use_amp": False, # Use auto mixed precision for optimization
    "amp_dtype": torch.float16 # will use amp_dtype only when use_amp is set to True
}

DATA_CONFIG = {
    "sample_size": 50, # Change if we want a different sample size
    "number_of_perturbations": 1 # Change if we want more perturbations
}