import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
# Attempt to import log_transform, handle if utils.utils might not exist in all contexts
try:
    from utils.utils import log_transform
except ImportError:
    # Fallback or placeholder if utils.utils or log_transform is not found
    # This is important if data_utils.py could be used independently of the main project structure
    def log_transform(x, eps=1e-6): # Basic placeholder
        return torch.log(x + eps) if isinstance(x, torch.Tensor) else np.log(x + eps)

import glob
import os
import random
from torch.utils.data import Sampler, Dataset
from typing import List

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

def variable_length_collate_fn(batch, max_seq_len_padding=None):
    """
    Collate function for variable-length sequences.
    Pads sequences to the maximum length in the batch or to max_seq_len_padding if provided.

    Args:
        batch (list): A list of dictionaries, where each dictionary is an output from
                      PrometheusDataset.__getitem__. Expected keys:
                      "times", "counts", "raw_times", "raw_counts", "sequence_length".
        max_seq_len_padding (int, optional): Maximum length to pad sequences to.
                                             If None, pads to the longest sequence in the batch.

    Returns:
        dict: A dictionary of batched and padded tensors.
              Keys: "times_padded", "counts_padded", "raw_times_padded",
                    "raw_counts_padded", "attention_mask", "sequence_lengths".
    """
    
    processed_batch_items = []
    for item_idx, item in enumerate(batch):
        current_raw_times = item["raw_times"] # These are torch tensors from __getitem__
        current_raw_counts = item["raw_counts"]
        current_seq_len = item["sequence_length"].item()

        if max_seq_len_padding is not None and current_seq_len > max_seq_len_padding:
            # Convert to numpy for easier manipulation in _adaptive_regroup
            # Ensure they are 1D arrays for the logic below
            raw_times_np = current_raw_times.cpu().numpy().flatten()
            raw_counts_np = current_raw_counts.cpu().numpy().flatten()
            
            num_merges_needed = current_seq_len - max_seq_len_padding
            
            for _ in range(num_merges_needed):
                if len(raw_times_np) <= 1: # Cannot merge further
                    break
                
                # Find pair with smallest time difference
                time_diffs = np.diff(raw_times_np)
                if len(time_diffs) == 0: # Should not happen if len(raw_times_np) > 1
                    break
                
                min_diff_idx = np.argmin(time_diffs)
                
                # Merge: new time is time of first event in pair, sum counts
                # The event at min_diff_idx is merged with event at min_diff_idx + 1
                # The time of the merged event is raw_times_np[min_diff_idx]
                # The count of the merged event is sum of counts
                raw_counts_np[min_diff_idx] = raw_counts_np[min_diff_idx] + raw_counts_np[min_diff_idx + 1]
                
                # Remove the second event of the merged pair
                raw_times_np = np.delete(raw_times_np, min_diff_idx + 1)
                raw_counts_np = np.delete(raw_counts_np, min_diff_idx + 1)

            # Update item with regrouped data
            item["raw_times"] = torch.from_numpy(raw_times_np).float().to(current_raw_times.device)
            item["raw_counts"] = torch.from_numpy(raw_counts_np).float().to(current_raw_counts.device)
            new_seq_len = len(raw_times_np)
            item["sequence_length"] = torch.tensor(new_seq_len, dtype=torch.long).to(item["sequence_length"].device)
            
            # Re-calculate log-transformed versions
            if new_seq_len > 0:
                # Assuming log_transform is available and handles tensor inputs
                # The log_transform from utils.utils expects torch tensors
                item["times"] = log_transform(item["raw_times"])
                item["counts"] = log_transform(item["raw_counts"])
            else:
                item["times"] = torch.empty(0, dtype=torch.float32).to(item["raw_times"].device)
                item["counts"] = torch.empty(0, dtype=torch.float32).to(item["raw_counts"].device)
        
        processed_batch_items.append(item)

    # Proceed with padding using the potentially regrouped items
    times_list = [item["times"] for item in processed_batch_items]
    counts_list = [item["counts"] for item in processed_batch_items]
    raw_times_list = [item["raw_times"] for item in processed_batch_items]
    raw_counts_list = [item["raw_counts"] for item in processed_batch_items]
    sequence_lengths = torch.tensor([item["sequence_length"].item() for item in processed_batch_items], dtype=torch.long)

    current_max_len = 0
    if sequence_lengths.numel() > 0 and sequence_lengths.max().item() > 0 :
        current_max_len = int(sequence_lengths.max().item())
    
    # Determine actual padding length for pad_sequence
    # If max_seq_len_padding is set, current_max_len is already <= max_seq_len_padding due to regrouping
    # If max_seq_len_padding is None, pad to the longest sequence in the batch.
    pad_to_len = current_max_len
    if max_seq_len_padding is not None:
        # If after regrouping, all sequences are shorter than max_seq_len_padding,
        # current_max_len will be that shorter length.
        # If some sequences were exactly max_seq_len_padding, current_max_len is max_seq_len_padding.
        # If all sequences were shorter than max_seq_len_padding to begin with, current_max_len is the max of those.
        # We should pad up to max_seq_len_padding if specified, unless all sequences are shorter.
        # The goal of max_seq_len_padding is to cap the length.
        # The regrouping ensures no sequence *exceeds* it.
        # The padding ensures all sequences *match* the length of the longest (up to this cap).
        if current_max_len == 0: # all sequences became empty or were empty
             pad_to_len = max_seq_len_padding if max_seq_len_padding > 0 else 0
        else:
             pad_to_len = current_max_len # Already capped by regrouping logic if needed

    # Pad sequences
    # pad_sequence expects a list of Tensors and pads them to the length of the longest Tensor in the list.
    # batch_first=True means the output will be (batch_size, seq_len, *)
    
    # Handle cases where all sequences in the batch might be empty after regrouping or initially
    # The `pad_to_len` variable now correctly reflects the target length for padding.
    
    # If pad_to_len is 0 (e.g., empty batch or all sequences became empty and no max_seq_len_padding)
    if pad_to_len == 0:
        batch_size = len(processed_batch_items)
        # Create tensors of shape (batch_size, 0)
        times_padded = torch.empty(batch_size, 0, dtype=torch.float32)
        counts_padded = torch.empty(batch_size, 0, dtype=torch.float32)
        raw_times_padded = torch.empty(batch_size, 0, dtype=torch.float32)
        raw_counts_padded = torch.empty(batch_size, 0, dtype=torch.float32)
        # Ensure tensors are on the correct device if possible (get from an example item if batch not empty)
        if batch_size > 0:
            example_device = processed_batch_items[0]["raw_times"].device
            times_padded = times_padded.to(example_device)
            counts_padded = counts_padded.to(example_device)
            raw_times_padded = raw_times_padded.to(example_device)
            raw_counts_padded = raw_counts_padded.to(example_device)
    else:
        # pad_sequence requires non-empty list of tensors, or tensors with at least one element if list has one tensor.
        # If a list is empty (e.g. times_list after all sequences became empty), pad_sequence might error.
        # We need to ensure that if pad_to_len > 0, we provide something pad_sequence can work with,
        # or construct the zero-padded tensors manually.
        
        # Helper to ensure list for pad_sequence is not entirely full of empty tensors if pad_to_len > 0
        def _prepare_for_pad(tensor_list, target_len, example_device):
            if target_len == 0: # If target length is 0, return list as is (will result in (B,0) tensors)
                 return [t if t.numel() > 0 else torch.empty(0, dtype=torch.float32).to(example_device) for t in tensor_list]

            # If target_len > 0, but all tensors in list are empty, pad_sequence might fail.
            # This case should be rare if current_max_len logic is correct.
            # For safety, if all are empty and target_len > 0, we might need to create zero tensors manually.
            # However, pad_sequence should handle lists of tensors, even if some are empty,
            # as long as the list itself is not empty.
            return tensor_list

        example_item_device = processed_batch_items[0]["raw_times"].device if len(processed_batch_items) > 0 else torch.device("cpu")

        times_padded = pad_sequence(_prepare_for_pad(times_list, pad_to_len, example_item_device), batch_first=True, padding_value=0.0)
        counts_padded = pad_sequence(_prepare_for_pad(counts_list, pad_to_len, example_item_device), batch_first=True, padding_value=0.0)
        raw_times_padded = pad_sequence(_prepare_for_pad(raw_times_list, pad_to_len, example_item_device), batch_first=True, padding_value=0.0)
        raw_counts_padded = pad_sequence(_prepare_for_pad(raw_counts_list, pad_to_len, example_item_device), batch_first=True, padding_value=0.0)

    # Create attention mask
    # The actual length used for padding by pad_sequence is now times_padded.size(1)
    # which should be equal to pad_to_len (unless pad_to_len was 0 and batch was empty)
    final_padded_len = times_padded.size(1)
    attention_mask = torch.zeros(len(processed_batch_items), final_padded_len, dtype=torch.bool)
    for i, length in enumerate(sequence_lengths):
        attention_mask[i, :length] = True
        
    return {
        "times_padded": times_padded,
        "counts_padded": counts_padded,
        "raw_times_padded": raw_times_padded,
        "raw_counts_padded": raw_counts_padded,
        "attention_mask": attention_mask,
        "sequence_lengths": sequence_lengths # Original lengths before padding (but after potential truncation)
    }