import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import glob
import os
import random
from torch.utils.data import Sampler, Dataset
from typing import List, Dict, Tuple
import pyarrow.parquet as pq

def get_file_names(data_dirs: List[str], 
                   ranges: List[List[int]], 
                   shuffle_files: bool = False) -> List[str]:
    """
    Get file names from directories within specified ranges.
    
    Args:
        data_dirs: List of directories to search for files
        ranges: List of [start, end] ranges for each directory
        
    Returns:
        List of file paths
    """
    filtered_files = []
    for i, directory in enumerate(data_dirs):
        all_files = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        if shuffle_files:
            random.shuffle(all_files)
        file_range = ranges[i]
        filtered_files.extend(
            all_files[file_range[0]:file_range[1]]
        )
    return sorted(filtered_files)

class ParquetFileSampler(Sampler):
    """
    Custom sampler for parquet files that respects file boundaries during batching.
    
    This sampler first selects files in random order, then for each file,
    it shuffles the indices and yields batches from that file before moving
    to the next file.
    """
    def __init__(self, 
                 data_source: Dataset, 
                 cumulative_lengths: np.ndarray, 
                 batch_size: int):
       super().__init__(data_source) # Call Sampler's __init__
       self.data_source = data_source
       self.cumulative_lengths = cumulative_lengths  # expects array starting with 0, then cumulative sums
       self.batch_size = batch_size

    def __iter__(self):
        n_files = len(self.cumulative_lengths) - 1
        file_order = np.random.permutation(n_files)
        
        for file_index in file_order:
            start_idx = self.cumulative_lengths[file_index]
            end_idx = self.cumulative_lengths[file_index + 1]
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            for i in range(0, len(indices), self.batch_size):
                yield from indices[i:i+self.batch_size].tolist()

    def __len__(self) -> int:
       return len(self.data_source)

def variable_length_collate_fn(batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate after all per-item shortening is done in __getitem__.
    Pads to the longest sequence in the mini-batch and builds attention-mask.
    Now expects 'charges_log_norm' and 'times_log_norm' (absolute times).
    """
    if not batch:
        return {}

    bsz = len(batch)
    # Determine device and dtype from the first item, assuming consistency
    # Handle empty batch items if they somehow occur
    first_valid_item = next((item for item in batch if item.get("charges_log_norm") is not None and item["charges_log_norm"].numel() > 0), None)
    if first_valid_item is None: # All items are empty or invalid
        # Fallback: create structure based on expected types if possible, or return minimal
        # This case should ideally be prevented by dataset filtering empty sequences if not desired
        # For now, let's assume at least one item might give a clue or we default.
        # If all items are truly empty as per __getitem__, they'll have empty tensors.
        # We'll proceed, and padding will handle it.
        # If even sensor_pos is missing, we might have issues.
        # Let's assume sensor_pos is always there.
        device = batch[0]["sensor_pos"].device
        dtype = batch[0]["sensor_pos"].dtype # Fallback dtype
        if "charges_log_norm" in batch[0]: # Check if key exists even if tensor is empty
            dtype = batch[0]["charges_log_norm"].dtype

    else:
        device = first_valid_item["charges_log_norm"].device
        dtype = first_valid_item["charges_log_norm"].dtype

    # Get sequence lengths from 'charges_log_norm' (or 'times_log_norm')
    seq_lens_list = [item["charges_log_norm"].numel() for item in batch]
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=device)
    max_len = int(seq_lens.max().item()) if bsz and seq_lens.numel() > 0 and seq_lens.max().item() > 0 else 0

    # ---------- pre-allocate padded buffers ----------
    charges_padded = torch.zeros(bsz, max_len, dtype=dtype, device=device)
    times_padded = torch.zeros(bsz, max_len, dtype=dtype, device=device) # For absolute times
    
    # True for padding, False for valid data
    attention_mask = torch.ones(bsz, max_len, dtype=torch.bool, device=device)
    
    sensor_pos_list = [item["sensor_pos"] for item in batch if "sensor_pos" in item]
    if sensor_pos_list:
        sensor_pos_batched = torch.stack(sensor_pos_list, dim=0)
    else:
        sensor_pos_batched = torch.empty(bsz, 3, dtype=dtype, device=device) # Fallback
    
    # ---------- write each sequence ----------
    for i, item in enumerate(batch):
        L = item["charges_log_norm"].numel() # Length of current sequence
        if L > 0:
            charges_padded[i, :L] = item["charges_log_norm"]
            times_padded[i, :L] = item["times_log_norm"]
            attention_mask[i, :L] = False  # Valid data positions
    
    return {
        "charges_log_norm_padded": charges_padded,
        "times_log_norm_padded": times_padded, # These are log-normalized ABSOLUTE times
        "attention_mask": attention_mask,      # True for padding
        "sensor_pos_batched": sensor_pos_batched,
        # The VAE's encode method will derive original_lengths from attention_mask
        # and other summary stats from these inputs.
    }