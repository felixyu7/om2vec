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
    return sorted(filtered_files)

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

def variable_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    max_seq_len_padding: int | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Optimised collate-fn for variable-length time-series.
    Keeps the original semantics (including the “merge-nearest-events”
    truncation rule) while trimming Python & allocation overhead.
    """

    bsz               = len(batch)
    device            = batch[0]["raw_times"].device
    dtype             = batch[0]["raw_times"].dtype
    processed_items   = []
    max_len_in_batch  = 0                       # longest sequence *after* any truncation

    # ------------------------------------------------------------------
    # 1. PER–ITEM PRE-PROCESS + OPTIONAL TRUNCATION
    # ------------------------------------------------------------------
    for item in batch:
        rt  = item["raw_times"]    # (L,)
        rc  = item["raw_counts"]
        L   = int(item["sequence_length"])

        # ---- optional “adaptive regroup” to respect max_seq_len_padding ----
        if max_seq_len_padding is not None and L > max_seq_len_padding:
            merges_needed = L - max_seq_len_padding
            # keep everything on torch (index-deletions are still cheap on 1-D tensors)
            for _ in range(merges_needed):
                # argmin() on diff is O(L), but L only shrinks; still faster
                # (and simpler) than a Python heap for the usual scale (<1k).
                idx = torch.argmin(torch.diff(rt))
                rc[idx] += rc[idx + 1]         # merge counts
                rt      = torch.cat((rt[:idx + 1],  rt[idx + 2:]))
                rc      = torch.cat((rc[:idx + 1],  rc[idx + 2:]))
                L      -= 1

        # ---- book-keeping & log-space tensors ----
        item["raw_times"]       = rt
        item["raw_counts"]      = rc
        item["times"]           = torch.log(rt + 1)
        item["counts"]          = torch.log(rc + 1)
        item["sequence_length"] = torch.tensor(L, dtype=torch.long, device=device)

        processed_items.append(item)
        max_len_in_batch = max(max_len_in_batch, L)

    # final target length (= longest sequence, optionally capped)
    pad_len = max_len_in_batch
    if max_seq_len_padding is not None:
        pad_len = min(pad_len, max_seq_len_padding)

    # ------------------------------------------------------------------
    # 2. PRE-ALLOCATE PADDED BUFFERS
    # ------------------------------------------------------------------
    shape               = (bsz, pad_len)
    zeros               = dict(size=shape, dtype=dtype, device=device)
    times_padded        = torch.zeros(**zeros)
    counts_padded       = torch.zeros_like(times_padded)
    raw_times_padded    = torch.zeros_like(times_padded)
    raw_counts_padded   = torch.zeros_like(times_padded)
    attention_mask      = torch.zeros(bsz, pad_len, dtype=torch.bool, device=device)
    seq_lens_tensor     = torch.empty(bsz, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 3. WRITE EACH SEQUENCE INTO THE PRE-ALLOCATED BUFFERS
    # ------------------------------------------------------------------
    for row, item in enumerate(processed_items):
        L = int(item["sequence_length"])
        if L == 0:                       # nothing to write
            seq_lens_tensor[row] = 0
            continue

        times_padded[row,       :L] = item["times"]
        counts_padded[row,      :L] = item["counts"]
        raw_times_padded[row,   :L] = item["raw_times"]
        raw_counts_padded[row,  :L] = item["raw_counts"]
        attention_mask[row,     :L] = True
        seq_lens_tensor[row]         = L

    return {
        "times_padded":     times_padded,
        "counts_padded":    counts_padded,
        "raw_times_padded": raw_times_padded,
        "raw_counts_padded": raw_counts_padded,
        "attention_mask":   attention_mask,
        "sequence_lengths": seq_lens_tensor,
    }