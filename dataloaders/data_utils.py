import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import glob
import os
import random
from torch.utils.data import Sampler, Dataset
from typing import List, Dict, Tuple
import pyarrow.parquet as pq

def get_file_names(data_dirs: List[str], ranges: List[List[int]], shuffle_files: bool = False) -> List[List[str]]:
    """
    Get file names from directories within specified ranges, grouped by directory.
    """
    files_by_folder = []

    for i, directory in enumerate(data_dirs):
        all_files_in_dir = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        
        file_range = ranges[i]
        selected_files_in_dir = all_files_in_dir[file_range[0]:file_range[1]]

        if not selected_files_in_dir:
            files_by_folder.append([])
            continue

        if shuffle_files:
            random.shuffle(selected_files_in_dir)
        
        files_by_folder.append(selected_files_in_dir)
            
    return files_by_folder

def variable_length_collate_fn(batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate after all per-item shortening is done in __getitem__.
    Pads to the longest sequence in the mini-batch and builds attention-mask.
    """
    if not batch:
        return {}

    bsz = len(batch)
    first_valid_item = next((item for item in batch if item.get("charges_log_norm") is not None and item["charges_log_norm"].numel() > 0), None)
    
    if first_valid_item is None:
        # Handle case where all items in the batch are empty
        sensor_pos_batched = torch.stack([item["sensor_pos"] for item in batch])
        return {
            "charges_log_norm_padded": torch.zeros(bsz, 0, dtype=torch.float32),
            "times_log_norm_padded": torch.zeros(bsz, 0, dtype=torch.float32),
            "attention_mask": torch.ones(bsz, 0, dtype=torch.bool),
            "sensor_pos_batched": sensor_pos_batched,
        }

    device = first_valid_item["charges_log_norm"].device
    dtype = first_valid_item["charges_log_norm"].dtype

    seq_lens_list = [item["charges_log_norm"].numel() for item in batch]
    max_len = max(seq_lens_list) if seq_lens_list else 0

    charges_padded = torch.zeros(bsz, max_len, dtype=dtype, device=device)
    times_padded = torch.zeros(bsz, max_len, dtype=dtype, device=device)
    attention_mask = torch.ones(bsz, max_len, dtype=torch.bool, device=device)
    
    sensor_pos_list = [item["sensor_pos"] for item in batch]
    sensor_pos_batched = torch.stack(sensor_pos_list, dim=0)
    
    for i, item in enumerate(batch):
        L = item["charges_log_norm"].numel()
        if L > 0:
            charges_padded[i, :L] = item["charges_log_norm"]
            times_padded[i, :L] = item["times_log_norm"]
            attention_mask[i, :L] = False
    
    return {
        "charges_log_norm_padded": charges_padded,
        "times_log_norm_padded": times_padded,
        "attention_mask": attention_mask,
        "sensor_pos_batched": sensor_pos_batched,
    }

class InterleavedFileBatchSampler(Sampler):
    """
    This is now a placeholder. The new dataset structure does not require a complex sampler.
    The standard DataLoader with shuffle=True will handle batching.
    """
    def __init__(self, data_source, batch_size, **kwargs):
        self.data_source = data_source
        self.batch_size = batch_size
        # kwargs are ignored, but kept for compatibility with old config.

    def __iter__(self):
        # This sampler is a no-op. The actual sampling is done by DataLoader's shuffle.
        # It yields indices from 0 to len(dataset)-1.
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)