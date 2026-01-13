from datasets import load_dataset

def get_sequences(sample_size: int):
    """Download the promoter_tata dataset and returns a sample of sequences

    Args:
        sample_size (int): Number of samples we want to use
    """
    print("Downloading dataset ...")
    dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        trust_remote_code=True
    ).filter(lambda x: x["task"] == "promoter_tata")
    
    sequences = dataset["train"]["sequence"]
    print(f"Dataset is loaded, we will be taking a sample of {sample_size}")
    return sequences[:sample_size]