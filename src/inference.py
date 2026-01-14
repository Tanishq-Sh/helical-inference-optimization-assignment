import torch
import numpy as np
import psutil
import logging
import time
import random
from helical.models.hyena_dna import HyenaDNA, HyenaDNAConfig
from src.config import OUTPUT_DIR, DEVICE, MODEL_CONFIG, DATA_CONFIG, RANDOM_SEED
from src.utils import add_pertubations, log_inference_profile
# logging.getLogger("helical.models.hyena_dna.model").setLevel(logging.WARNING)


# Download the model using the hyena_config
def load_hyena_model():
    """
    Loads the HyenaDNA model based on settings in config.py,
    moves it to the correct device, and returns the model object.
    """
    hyena_config = HyenaDNAConfig(
        model_name = MODEL_CONFIG["model_name"], # HyenaDNA models can be used here
    )
    model = HyenaDNA(configurer=hyena_config)

    # move model to use GPU if possible
    if DEVICE == "cuda":
        model.model.to(DEVICE)
        
    print("Model loaded successfully!")
    
    return model

def run_hyena_inferencing(model, sequences_to_process: list):

    pertubation_embeddings = []
    latencies = []

    # Inference on pertubations
    start = time.time()
    start_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    BATCH_SIZE = 32
    overall_start = time.time()

    for i in range(0, DATA_CONFIG["sample_size"], BATCH_SIZE):
        t_loop_in = time.time()
        raw_tokens = model.process_data(sequences_to_process[i:i + BATCH_SIZE])
        input_ids_tensor = torch.tensor(raw_tokens["input_ids"]).to(DEVICE)

        with torch.no_grad():
            outputs = model.model(input_ids=input_ids_tensor)
            embeddings = outputs

        t_loop_out = time.time()
        latencies.append(t_loop_out - t_loop_in)
        
    if isinstance(embeddings, torch.Tensor):
        pertubation_embeddings.append(embeddings)

    total_time = time.time() - overall_start

    # Call the logging function from utils
    log_inference_profile(
        total_time=total_time,
        latencies=latencies,
        num_samples=len(sequences_to_process),
        start_rss_mb=start_rss
    )

    return torch.cat(pertubation_embeddings, dim=0)

        




