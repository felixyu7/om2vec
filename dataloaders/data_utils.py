import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import glob
import os
import random
from torch.utils.data import Sampler, Dataset
from typing import List, Dict, Tuple # Added Tuple
import pyarrow.parquet as pq # Added import

def get_file_names(data_dirs: List[str], ranges: List[List[int]], shuffle_files: bool = False) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Get file names from directories within specified ranges, grouped by directory.
    Also returns the number of events (rows) for each file.

    Args:
        data_dirs: List of directories to search for files
        ranges: List of [start, end] ranges for files in each directory
        shuffle_files: Whether to shuffle files within each directory's list

    Returns:
        A tuple containing:
            - files_by_folder: List of lists of file paths, one inner list per directory.
            - events_per_file_by_folder: List of lists of event counts, mirroring files_by_folder.
    """
    files_by_folder = []
    events_per_file_by_folder = []

    for i, directory in enumerate(data_dirs):
        all_files_in_dir = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        
        # Apply range selection
        file_range = ranges[i]
        selected_files_in_dir = all_files_in_dir[file_range[0]:file_range[1]]

        if not selected_files_in_dir:
            files_by_folder.append([])
            events_per_file_by_folder.append([])
            continue

        current_folder_files = []
        current_folder_event_counts = []

        # Prepare list of (file_path, event_count) for shuffling if needed
        file_event_pairs = []
        for f_path in selected_files_in_dir:
            try:
                pf = pq.ParquetFile(f_path)
                event_count = pf.metadata.num_rows
                file_event_pairs.append((f_path, event_count))
            except Exception as e:
                # print(f"Warning: Could not read metadata for {f_path}: {e}. Skipping file.")
                # Optionally, append with 0 events or handle differently
                # For now, we skip files that error out during metadata read.
                continue # Skip this file
        
        if not file_event_pairs: # All selected files in dir failed to read metadata
            files_by_folder.append([])
            events_per_file_by_folder.append([])
            continue

        if shuffle_files:
            random.shuffle(file_event_pairs)
        
        # Unzip after potential shuffle
        current_folder_files, current_folder_event_counts = zip(*file_event_pairs)
        
        files_by_folder.append(list(current_folder_files))
        events_per_file_by_folder.append(list(current_folder_event_counts))
            
    return files_by_folder, events_per_file_by_folder

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

class MultiFolderBatchSampler:
    """
    Batch Sampler that draws a configurable number of samples from each folder to form a batch.
    Ensures mixing of events from different data sources (folders).
    Yields batches of (folder_idx, event_idx_in_folder) tuples.
    """
    def __init__(self,
                 dataset, # Expects the modified PrometheusTimeSeriesDataset
                 batch_size: int,
                 shuffle_intra_batch: bool = True,
                 shuffle_folders_epoch: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_intra_batch = shuffle_intra_batch
        self.shuffle_folders_epoch = shuffle_folders_epoch
        self.num_folders = self.dataset.num_folders

        if self.num_folders == 0 or self.batch_size == 0:
            self.samples_per_folder_exact = []
            self.total_samples_per_batch = 0
            self.num_batches = 0
            return

        base_samples_per_folder = self.batch_size // self.num_folders
        remainder_samples = self.batch_size % self.num_folders
        
        self.samples_per_folder_exact = [base_samples_per_folder] * self.num_folders
        for i in range(remainder_samples):
            self.samples_per_folder_exact[i % self.num_folders] += 1
        
        self.total_samples_per_batch = sum(self.samples_per_folder_exact)
        
        if self.total_samples_per_batch == 0: # Handles batch_size = 0 or all folders empty leading to 0 samples
            self.num_batches = 0
            return

        num_batches_per_folder = []
        for f_idx in range(self.num_folders):
            if self.dataset.folder_total_events[f_idx] == 0:
                num_batches_per_folder.append(float('inf'))
                continue
            if self.samples_per_folder_exact[f_idx] > 0:
                num_batches_per_folder.append(self.dataset.folder_total_events[f_idx] // self.samples_per_folder_exact[f_idx])
            else: # This folder contributes 0 samples per batch
                num_batches_per_folder.append(float('inf'))

        self.num_batches = min(b for b in num_batches_per_folder if b != float('inf')) if any(b != float('inf') for b in num_batches_per_folder) else 0
        if self.num_batches == float('inf'): # All folders were empty or assigned 0 samples
             self.num_batches = 0


    def __iter__(self):
        if self.num_folders == 0 or self.num_batches == 0 or self.total_samples_per_batch == 0:
            return iter([])

        folder_event_indices_lists = []
        for f_idx in range(self.num_folders):
            if self.dataset.folder_total_events[f_idx] > 0:
                indices = list(range(self.dataset.folder_total_events[f_idx]))
                if self.shuffle_folders_epoch:
                    random.shuffle(indices)
                folder_event_indices_lists.append(indices)
            else:
                folder_event_indices_lists.append([])

        current_pos_in_folder_indices = [0] * self.num_folders

        for _ in range(self.num_batches):
            current_batch_tuples = []
            possible_to_form_batch = True
            for f_idx in range(self.num_folders):
                num_samples_to_take = self.samples_per_folder_exact[f_idx]
                if num_samples_to_take == 0:
                    continue

                start_ptr = current_pos_in_folder_indices[f_idx]
                end_ptr = start_ptr + num_samples_to_take

                if end_ptr > len(folder_event_indices_lists[f_idx]):
                    possible_to_form_batch = False
                    break
                
                for i in range(start_ptr, end_ptr):
                    event_idx_in_folder = folder_event_indices_lists[f_idx][i]
                    current_batch_tuples.append((f_idx, event_idx_in_folder))
                
                current_pos_in_folder_indices[f_idx] = end_ptr

            if possible_to_form_batch and len(current_batch_tuples) == self.total_samples_per_batch:
                if self.shuffle_intra_batch:
                    random.shuffle(current_batch_tuples)
                yield current_batch_tuples
            elif not possible_to_form_batch:
                break
            elif len(current_batch_tuples) != self.total_samples_per_batch and self.total_samples_per_batch > 0 :
                # This case implies an issue, e.g. a folder was expected to contribute but couldn't.
                # Or total_samples_per_batch was miscalculated relative to actual contributions.
                # For safety, break.
                break


    def __len__(self) -> int:
        return self.num_batches