import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import glob
import os
import random
from torch.utils.data import Sampler, Dataset
from typing import List, Dict

def get_file_names(data_dirs: List[str], ranges: List[List[int]], shuffle_files: bool = False) -> List[str]:
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
    if shuffle_files:
        random.shuffle(filtered_files)
    return filtered_files

class ParquetFileSampler(Sampler):
    """
    Custom sampler for parquet files that respects file boundaries during batching.
    
    This sampler first selects files in random order, then for each file,
    it shuffles the indices and yields batches from that file before moving
    to the next file.
    """
    def __init__(self, data_source: Dataset, cumulative_lengths: np.ndarray, batch_size: int):
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
    """

    bsz      = len(batch)
    device   = batch[0]["times"].device
    dtype    = batch[0]["times"].dtype

    seq_lens = torch.tensor([it["sequence_length"].item() for it in batch],
                            dtype=torch.long, device=device)
    max_len  = int(seq_lens.max().item()) if bsz else 0

    # ---------- pre-allocate padded buffers ----------
    shape      = (bsz, max_len)
    zeros_f32  = dict(size=shape, dtype=dtype, device=device)

    times_pad   = torch.zeros(**zeros_f32)
    counts_pad  = torch.zeros_like(times_pad)
    raw_t_pad   = torch.zeros_like(times_pad)
    raw_c_pad   = torch.zeros_like(times_pad)
    attn_mask   = torch.zeros(bsz, max_len, dtype=torch.bool, device=device)

    # ---------- write each sequence ----------
    for row, item in enumerate(batch):
        L = int(item["sequence_length"])
        if L == 0:
            continue
        times_pad[row,  :L] = item["times"]
        counts_pad[row, :L] = item["counts"]
        raw_t_pad[row,  :L] = item["raw_times"]
        raw_c_pad[row,  :L] = item["raw_counts"]
        attn_mask[row,  :L] = True

    return {
        "times_padded":     times_pad,
        "counts_padded":    counts_pad,
        "raw_times_padded": raw_t_pad,
        "raw_counts_padded": raw_c_pad,
        "attention_mask":   attn_mask,
        "sequence_lengths": seq_lens,
    }