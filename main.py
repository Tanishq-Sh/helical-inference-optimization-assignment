import logging
import time
from src import config
from src.data_processing import get_sequences
from src.inference import load_hyena_model, run_hyena_inferencing
from src.utils import add_pertubations
import os
import numpy as np

log = logging.getLogger(__name__)

def setup_logging(time_signature: str):
    # This function now configures the ROOT logger, which all child loggers inherit from.
    # The implementation can remain the same as the previous suggestion.
    os.makedirs(os.path.join(config.OUTPUT_DIR, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, "logs", f"inference_log_{time_signature}.txt")),
            logging.StreamHandler()
        ],
        force=True 
    )
    log.info("Logging configured.") # Use the named logger here too

def main(time_signature: str):
        
    logging.info(f"Using device: {config.DEVICE}")
    
    sequences = get_sequences(config.DATA_CONFIG["sample_size"])

    perturbed_sequences = []

    for sequence in sequences:
        perturbed_sequences.append(add_pertubations(sequence, num_of_pertubations=1))
    
    perturbed_sequences = [add_pertubations(seq, num_of_pertubations=config.DATA_CONFIG["number_of_perturbations"]) for seq in sequences]
    
    logging.info("Loading Hyena model...")
    model = load_hyena_model()
    
    logging.info("Starting inference run on original sequences...")
    original_embeddings = run_hyena_inferencing(model, sequences)
    
    logging.info("Starting inference run on perturbed sequences...")
    perturbed_embeddings = run_hyena_inferencing(model, perturbed_sequences)
    
    logging.info(f"Successfully generated original embeddings of shape: {original_embeddings.shape}")
    logging.info(f"Successfully generated perturbed embeddings of shape: {perturbed_embeddings.shape}")
    
    original_embedding_array = np.stack(original_embeddings)
    perturbed_embedding_array = np.stack(perturbed_embeddings)

    os.makedirs(os.path.join(config.OUTPUT_DIR, "embeddings"), exist_ok=True) # Create output directory if it doesn't exist
    
    save_ts = time.time()
    np.save(os.path.join(config.OUTPUT_DIR, "embeddings", f"original_embedding_{time_signature}.npy"), original_embedding_array)
    np.save(os.path.join(config.OUTPUT_DIR, "embeddings", f"perturbed_embedding_{time_signature}.npy"), original_embedding_array)
    
if __name__ == "__main__":
    script_start_time = time.strftime('%Y%m%d-%H%M%S')
    setup_logging(time_signature=script_start_time)
    try:
        main(time_signature=script_start_time)
    finally:
        # This ensures that all buffered log records are written to the file
        # before the program exits, even if an error occurs.
        logging.shutdown()