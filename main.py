import logging
import time
from src import config
from src.data_processing import get_sequences
from src.inference import load_hyena_model, run_hyena_inferencing
from src.utils import add_pertubations
import os

log = logging.getLogger(__name__)

def setup_logging():
    # This function now configures the ROOT logger, which all child loggers inherit from.
    # The implementation can remain the same as the previous suggestion.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, f"inference_log_{time.strftime('%Y%m%d-%H%M%S')}.txt")),
            logging.StreamHandler()
        ],
        force=True 
    )
    log.info("Logging configured.") # Use the named logger here too

def main():
        
    logging.info(f"Using device: {config.DEVICE}")
    
    sequences = get_sequences(config.DATA_CONFIG["sample_size"])

    perturbed_sequences = []

    for sequence in sequences:
        perturbed_sequences.append(add_pertubations(sequence, num_of_pertubations=1))
    
    perturbed_sequences = [add_pertubations(seq, num_of_pertubations=config.DATA_CONFIG["number_of_perturbations"]) for seq in sequences]
    
    logging.info("Loading Hyena model...")
    model = load_hyena_model()
    
    logging.info("Starting inference run...")
    embeddings = run_hyena_inferencing(model, perturbed_sequences)
    
    logging.info(f"Successfully generated embeddings of shape: {embeddings.shape}")
    
if __name__ == "__main__":
    setup_logging()
    try:
        main()
    finally:
        # This ensures that all buffered log records are written to the file
        # before the program exits, even if an error occurs.
        logging.shutdown()