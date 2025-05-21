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
    
    # Handle eos_target (pad to max_len)
    eos_targets = [item["eos_target"] for item in batch]
    eos_target_padded = torch.zeros(bsz, max_len, dtype=torch.float32, device=device)
    for row, eos in enumerate(eos_targets):
        L = eos.shape[0]
        eos_target_padded[row, :L] = eos

    # Handle sensor_pos (fixed size per item, so just stack)
    # Assuming sensor_pos is a 1D tensor of 3 elements
    sensor_pos_list = [item["sensor_pos"] for item in batch if "sensor_pos" in item]
    if sensor_pos_list:
        sensor_pos_batched = torch.stack(sensor_pos_list, dim=0)
    else:
        # Fallback if no sensor_pos found
        sensor_pos_batched = torch.empty(bsz, 3, dtype=dtype, device=device)

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
        "eos_target_padded": eos_target_padded,
        "sensor_pos_batched": sensor_pos_batched,
    }

class InterleavedFileBatchSampler:
    """
    Batch Sampler that interleaves data from multiple folders by processing one file
    from each folder at a time. A set of "active files" (one per folder) is used
    to generate batches until one of these files is exhausted. Then, the sampler
    advances to the next file in the folder(s) whose file(s) were exhausted.

    Yields batches of (folder_idx, event_idx_in_folder) tuples.
    """
    def __init__(self,
                 dataset,  # Expects PrometheusTimeSeriesDataset
                 batch_size: int,
                 shuffle_intra_batch: bool = True,
                 shuffle_files_within_folder: bool = True): # Added to control if files within a folder are shuffled
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_intra_batch = shuffle_intra_batch
        self.num_folders = self.dataset.num_folders

        if self.num_folders == 0 or self.batch_size == 0:
            self._num_batches = 0
            return

        # Determine how many samples to draw from each folder per batch
        base_samples_per_folder = self.batch_size // self.num_folders
        remainder_samples = self.batch_size % self.num_folders
        self.samples_per_folder_in_batch = [base_samples_per_folder] * self.num_folders
        for i in range(remainder_samples):
            self.samples_per_folder_in_batch[i % self.num_folders] += 1
        
        self.actual_batch_size = sum(self.samples_per_folder_in_batch)

        # Store file paths and their event counts, potentially shuffled per folder
        self.files_by_folder = []
        self.events_in_files_by_folder = []

        for f_idx in range(self.num_folders):
            folder_files = list(self.dataset.files_by_folder[f_idx]) # Make a copy
            folder_event_counts = list(self.dataset.events_per_file_by_folder[f_idx]) # Make a copy
            
            if shuffle_files_within_folder and folder_files:
                paired = list(zip(folder_files, folder_event_counts))
                random.shuffle(paired)
                shuffled_f, shuffled_e = zip(*paired)
                self.files_by_folder.append(list(shuffled_f))
                self.events_in_files_by_folder.append(list(shuffled_e))
            else:
                self.files_by_folder.append(folder_files)
                self.events_in_files_by_folder.append(folder_event_counts)
        
        self._num_batches = self._calculate_total_batches()

    def _calculate_total_batches(self) -> int:
        if self.num_folders == 0 or self.actual_batch_size == 0:
            return 0

        total_batches = 0
        current_file_indices = [0] * self.num_folders # Index of the current file being processed in each folder
        
        # Create a copy of event counts for simulation
        sim_events_in_files = [[count for count in folder_counts] for folder_counts in self.events_in_files_by_folder]

        while True:
            active_files_exist = False
            min_batches_from_current_set = float('inf')
            
            # Check if there's an active file for each folder that contributes to a batch
            # and determine how many batches this set can produce
            possible_to_form_set = True
            for f_idx in range(self.num_folders):
                if self.samples_per_folder_in_batch[f_idx] == 0: # This folder doesn't contribute
                    continue

                if current_file_indices[f_idx] >= len(sim_events_in_files[f_idx]):
                    possible_to_form_set = False # This folder is out of files
                    break
                
                active_files_exist = True
                num_events_in_active_file = sim_events_in_files[f_idx][current_file_indices[f_idx]]
                batches_from_this_file = num_events_in_active_file // self.samples_per_folder_in_batch[f_idx]
                min_batches_from_current_set = min(min_batches_from_current_set, batches_from_this_file)

            if not possible_to_form_set or not active_files_exist or min_batches_from_current_set == float('inf'):
                break # No more complete sets of active files or a folder is exhausted

            if min_batches_from_current_set == 0 and active_files_exist : # A file is too small for even one contribution
                 # Advance the file index for all folders that contributed to this problematic set
                 # to avoid infinite loop if a file is smaller than its per-batch contribution.
                advanced_once = False
                for f_idx in range(self.num_folders):
                    if self.samples_per_folder_in_batch[f_idx] > 0 and \
                       current_file_indices[f_idx] < len(sim_events_in_files[f_idx]) and \
                       sim_events_in_files[f_idx][current_file_indices[f_idx]] < self.samples_per_folder_in_batch[f_idx]:
                        current_file_indices[f_idx] += 1
                        advanced_once = True
                if not advanced_once: # Should not happen if min_batches_from_current_set was 0
                    break
                continue


            total_batches += min_batches_from_current_set
            
            # Consume events and advance file indices for exhausted files
            for f_idx in range(self.num_folders):
                if self.samples_per_folder_in_batch[f_idx] == 0:
                    continue
                if current_file_indices[f_idx] < len(sim_events_in_files[f_idx]):
                    events_to_consume = min_batches_from_current_set * self.samples_per_folder_in_batch[f_idx]
                    # This subtraction is just for simulation, original counts are preserved in self.events_in_files_by_folder
                    # The actual check for exhaustion in __iter__ will be based on offsets.
                    # For calculation, we just need to know when to advance the file pointer.
                    sim_events_in_files[f_idx][current_file_indices[f_idx]] -= events_to_consume
                    if sim_events_in_files[f_idx][current_file_indices[f_idx]] < self.samples_per_folder_in_batch[f_idx] : # or effectively zero
                        current_file_indices[f_idx] += 1
        return total_batches

    def __iter__(self):
        if self.num_folders == 0 or self._num_batches == 0 or self.actual_batch_size == 0:
            return iter([])

        # Pointers to the current file index within each folder's list of files
        current_file_indices_in_folder = [0] * self.num_folders
        # Pointers to the current event offset *within the currently active file* of each folder
        current_event_offsets_in_active_files = [0] * self.num_folders
        
        # Cumulative event counts for files *within each folder* (from dataset, but using potentially shuffled file order)
        # This helps map a local event_idx_in_folder (across concatenated files of that folder) to a specific file and offset.
        # We need to reconstruct this based on self.events_in_files_by_folder
        cumulative_lengths_by_folder_shuffled_files = []
        for f_idx in range(self.num_folders):
            if self.events_in_files_by_folder[f_idx]:
                cumulative_lengths_by_folder_shuffled_files.append(
                    np.concatenate(([0], np.cumsum(np.array(self.events_in_files_by_folder[f_idx], dtype=np.int64))))
                )
            else:
                cumulative_lengths_by_folder_shuffled_files.append(np.array([0], dtype=np.int64))


        for _ in range(self._num_batches): # Iterate for the pre-calculated number of batches
            batch_event_tuples = []
            
            # Determine the number of batches that can be formed from the current set of active files
            # This logic is implicitly handled by _calculate_total_batches ensuring we don't over-iterate.
            # Here, we just need to pick events for *one* batch.

            possible_to_form_this_batch = True
            for f_idx in range(self.num_folders):
                if self.samples_per_folder_in_batch[f_idx] == 0:
                    continue

                # Check if current folder has enough files left
                if current_file_indices_in_folder[f_idx] >= len(self.files_by_folder[f_idx]):
                    possible_to_form_this_batch = False
                    break
                
                active_file_total_events = self.events_in_files_by_folder[f_idx][current_file_indices_in_folder[f_idx]]
                
                # Check if current active file has enough remaining events for this batch
                if current_event_offsets_in_active_files[f_idx] + self.samples_per_folder_in_batch[f_idx] > active_file_total_events:
                    possible_to_form_this_batch = False # This specific file is exhausted for this batch round
                    # This situation should ideally be caught by _calculate_total_batches logic,
                    # leading to file advancement before this point. If hit, it implies a mismatch or edge case.
                    break

            if not possible_to_form_this_batch:
                # This signifies an issue with batch calculation or state,
                # or that we need to advance files. The outer loop of _calculate_total_batches
                # should prevent needing to advance files *mid-batch construction* here.
                # For safety, if we can't form a full batch as expected by _num_batches, we stop.
                # print("Warning: Could not form a full batch as expected. Ending iteration.")
                break

            for f_idx in range(self.num_folders):
                num_samples_to_take = self.samples_per_folder_in_batch[f_idx]
                if num_samples_to_take == 0:
                    continue

                # Base event index in this folder, considering all previous files in this folder
                base_event_idx_in_folder = cumulative_lengths_by_folder_shuffled_files[f_idx][current_file_indices_in_folder[f_idx]]
                
                for i in range(num_samples_to_take):
                    # event_idx_in_current_file = current_event_offsets_in_active_files[f_idx] + i
                    # The dataset's __getitem__ expects an overall event index within the folder
                    event_idx_in_folder_for_dataset = base_event_idx_in_folder + current_event_offsets_in_active_files[f_idx] + i
                    batch_event_tuples.append((f_idx, event_idx_in_folder_for_dataset))
                
                current_event_offsets_in_active_files[f_idx] += num_samples_to_take

            if self.shuffle_intra_batch:
                random.shuffle(batch_event_tuples)
            
            if len(batch_event_tuples) == self.actual_batch_size:
                 yield batch_event_tuples
            else:
                # print(f"Warning: Batch size mismatch. Expected {self.actual_batch_size}, got {len(batch_event_tuples)}. Ending.")
                break # Safety break

            # After yielding a batch, check if any active file is now exhausted
            # and advance to the next file for those folders.
            needs_file_advancement = False
            for f_idx in range(self.num_folders):
                if self.samples_per_folder_in_batch[f_idx] == 0:
                    continue
                if current_file_indices_in_folder[f_idx] >= len(self.files_by_folder[f_idx]): # Already out of files
                    continue

                active_file_total_events = self.events_in_files_by_folder[f_idx][current_file_indices_in_folder[f_idx]]
                # If the offset + next sample requirement exceeds current file, it's exhausted for next round
                if current_event_offsets_in_active_files[f_idx] + self.samples_per_folder_in_batch[f_idx] > active_file_total_events \
                   or current_event_offsets_in_active_files[f_idx] >= active_file_total_events : # Fully exhausted
                    current_file_indices_in_folder[f_idx] += 1
                    current_event_offsets_in_active_files[f_idx] = 0
                    needs_file_advancement = True # Signal that at least one file was advanced

            # If no files were advanced, it means all active files still have data for the next batch.
            # If files *were* advanced, the next iteration of the loop will use the new set of active files.
            # The overall loop is controlled by `_num_batches`, which should account for these advancements.

    def __len__(self) -> int:
        return self._num_batches