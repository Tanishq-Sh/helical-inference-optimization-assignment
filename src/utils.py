import numpy as np
import random
import psutil
import torch
import logging
from src.config import DATA_CONFIG, MODEL_CONFIG, DEVICE


def add_pertubations(sequence_string, num_of_pertubations):
  """adds pertubations to a sequence of nucleotides"""
  nucleotides = ["A", "G", "T", "C"]
  length = len(sequence_string)
  seq_list = list(sequence_string)

  for _ in range(num_of_pertubations):
    random_idx = np.random.randint(0, length - 1)

    original_nucleotide = seq_list[random_idx]
    possible_pertubations = [n for n in nucleotides if n != original_nucleotide]
    new_nucleotide = random.choice(possible_pertubations)

    # apply the pertubation to mutate
    seq_list[random_idx] = new_nucleotide

  # return perturbed sequence
  return "".join(seq_list)


def log_inference_profile(
  total_time: float,
  latencies: list,
  num_samples: int,
  start_rss_mb: float
):
  """
  Calculates and logs inference related metrics of the run
  """
  avg_latency = np.mean(latencies) * 1000 # to convert in ms
  throughput = DATA_CONFIG["sample_size"]/total_time
  end_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
  
  # get GPU memory if running on GPU
  peak_gpu_mb = 0
  if DEVICE == "cuda" and torch.cuda.is_available():
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
  log_line = f"""
------------ Inference Profile ------------
Device:                 {DEVICE.upper()}
Total Samples:          {num_samples}
Batch Size:             {DATA_CONFIG["BATCH_SIZE"]}
---
Total Time:             {total_time:.2f} s
Throughput:             {throughput:.2f} samples/s
Avg. Latency / Batch:   {avg_latency:.2f} ms
---
CPU RAM Usage:          {end_rss_mb - start_rss_mb:.2f} MB
Peak GPU Memory:        {peak_gpu_mb:.2f} MB
---------------------------------------------------
"""
  logging.info(log_line)
  # Reset peak memory stats for the next run if needed
  if DEVICE == "cuda" and torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()
  
