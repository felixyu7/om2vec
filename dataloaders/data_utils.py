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

class FileBatchSampler(Sampler):
    """
    Custom sampler that shuffles files and then yields batches of indices from each file.
    This is I/O efficient as it reads one file at a time.
    """
    def __init__(self, data_source: Dataset, batch_size: int, shuffle: bool = True):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cumulative_lengths = data_source.cumulative_lengths
        self.total_events = data_source.dataset_size

    def __iter__(self):
        n_files = len(self.cumulative_lengths) - 1
        file_indices = np.random.permutation(n_files) if self.shuffle else np.arange(n_files)

        for file_idx in file_indices:
            start_event_idx = self.cumulative_lengths[file_idx]
            end_event_idx = self.cumulative_lengths[file_idx + 1]
            
            event_indices_in_file = np.arange(start_event_idx, end_event_idx)
            if self.shuffle:
                np.random.shuffle(event_indices_in_file)
            
            for i in range(0, len(event_indices_in_file), self.batch_size):
                yield event_indices_in_file[i:i+self.batch_size].tolist()

    def __len__(self) -> int:
        # This is an estimate, as the last batch of each file might be smaller.
        # A more accurate length would be sum(ceil(file_size / batch_size))
        return (self.total_events + self.batch_size - 1) // self.batch_size

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

# The InterleavedFileBatchSampler has been removed and replaced by the simpler FileBatchSampler.