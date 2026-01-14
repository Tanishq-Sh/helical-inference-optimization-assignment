# HyenaDNA Inference Pipeline

This project runs inference using the HyenaDNA model on a sample of DNA sequences. It profiles the performance and saves the resulting embeddings and logs for each run.

## Project Structure

```
.
├── main.py              # Main script to run everything
├── environment.yml      # Conda environment setup
├── src/                 # Source code
│   ├── config.py        # All run settings are here
│   ├── data_processing.py
│   ├── inference.py
│   └── utils.py
├── collab_notebooks/    # Jupyter notebooks for experimentation and analysis
└── output/              # All results are saved here
    ├── embeddings/
    └── logs/
```

## How to Run

1.  **Set up Environment**:
    ```bash
    # Create the conda environment
    conda env create -f environment.yml

    # Activate it
    conda activate helical-env
    ```

2.  **Configure (Optional)**:
    Edit `src/config.py` to change the batch size, sample size, etc.

3.  **Run**:
    ```bash
    python main.py
    ```
    Logs and embeddings will be saved in the `output` directory, named with a timestamp.

## Notebook for Experimentation

You can use the Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vnviiip2vLAJGDBmxU1t2EF5gmsb3p6H?authuser=1)