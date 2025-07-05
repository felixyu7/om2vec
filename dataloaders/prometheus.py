import os
import glob
import torch
import numpy as np
import math
import pytorch_lightning as pl
import awkward as ak
import pyarrow.parquet as pq
from typing import Dict, Tuple, List # Added Tuple, List
from collections import OrderedDict # For FIFO cache

from .data_utils import get_file_names, FileBatchSampler, variable_length_collate_fn
from functools import partial

class PrometheusTimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for Prometheus dataset.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def prepare_data(self):
        """Called only once and on 1 GPU."""
        pass
    
    def setup(self, stage=None):
        """
        Called on each GPU separately - shards data to all GPUs.
        Sets up train and validation datasets.
        """
        if self.cfg['training']:
            train_files_by_folder, train_events_per_file_by_folder = get_file_names(
                self.cfg['data_options']['train_data_files'],
                self.cfg['data_options']['train_data_file_ranges'],
                self.cfg['data_options']['shuffle_files']
            )
            # Flatten the lists of files and event counts
            all_train_files = [file for folder in train_files_by_folder for file in folder]
            all_train_event_counts = [count for folder in train_events_per_file_by_folder for count in folder]
            
            self.train_dataset = PrometheusTimeSeriesDataset(
                all_train_files,
                all_train_event_counts,
                self.cfg
            )
            
        valid_files_by_folder, valid_events_per_file_by_folder = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            shuffle_files=False # Validation files are typically not shuffled
        )
        # Flatten the lists for validation set
        all_valid_files = [file for folder in valid_files_by_folder for file in folder]
        all_valid_event_counts = [count for folder in valid_events_per_file_by_folder for count in folder]

        self.valid_dataset = PrometheusTimeSeriesDataset(
            all_valid_files,
            all_valid_event_counts,
            self.cfg
        )

    def train_dataloader(self):
        """Returns the training dataloader."""
        batch_sampler = FileBatchSampler(
            data_source=self.train_dataset,
            batch_size=self.cfg['training_options']['batch_size'],
            shuffle=True
        )
        collate_fn_with_padding = partial(variable_length_collate_fn)

        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=collate_fn_with_padding,
                                           pin_memory=True,
                                           persistent_workers=self.cfg['training_options']['num_workers'] > 0,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def val_dataloader(self):
        """Returns the validation dataloader."""
        batch_sampler = FileBatchSampler(
            data_source=self.valid_dataset,
            batch_size=self.cfg['training_options']['batch_size'],
            shuffle=False
        )
        collate_fn_with_padding = partial(variable_length_collate_fn)

        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=collate_fn_with_padding,
                                           pin_memory=True,
                                           persistent_workers=self.cfg['training_options']['num_workers'] > 0,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        return self.val_dataloader()

class PrometheusTimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset for Prometheus data. Loads data from a list of parquet files.
    """
    def __init__(self,
                 files: List[str],
                 events_per_file: List[int],
                 cfg: Dict):
        self.files = files
        self.events_per_file = events_per_file
        self.cfg = cfg
        self.grouping_window_ns = self.cfg['data_options'].get('grouping_window_ns', 2.0)
        self.max_seq_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None)
        
        self.dataset_size = sum(self.events_per_file)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(np.array(self.events_per_file, dtype=np.int64))))

        # Simple FIFO cache for loaded Parquet data
        self.cache_size = self.cfg['data_options'].get('cache_size', 5)
        self.data_cache = OrderedDict()

    def __len__(self):
        return self.dataset_size
    
    def _get_file_path_and_index_in_file(self, global_event_idx: int) -> Tuple[str, int]:
        """
        Maps a global event index to a specific file and the event's index within that file.
        """
        file_idx = int(np.searchsorted(self.cumulative_lengths, global_event_idx + 1) - 1)
        event_idx_in_file = int(global_event_idx - self.cumulative_lengths[file_idx])
        file_path = self.files[file_idx]
        return file_path, event_idx_in_file

    def _load_data_from_file(self, file_path: str):
        """Loads data from a parquet file, using a cache."""
        if file_path in self.data_cache:
            self.data_cache.move_to_end(file_path)
            return self.data_cache[file_path]
        
        data = ak.from_parquet(file_path)
        if len(self.data_cache) >= self.cache_size:
            self.data_cache.popitem(last=False)
        self.data_cache[file_path] = data
        return data

    def __getitem__(self, global_event_idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one training example.
        """
        file_path, event_idx_in_file = self._get_file_path_and_index_in_file(global_event_idx)
        
        ak_data_for_file = self._load_data_from_file(file_path)
        row = ak_data_for_file[event_idx_in_file]

        # ------------------------------------------------ raw hit times
        raw_t = np.asarray(row['hits_t'], dtype=np.float32)
        
        # Extract sensor position
        sensor_pos_x = float(row['sensor_pos_x']) if 'sensor_pos_x' in row.fields else 0.0
        sensor_pos_y = float(row['sensor_pos_y']) if 'sensor_pos_y' in row.fields else 0.0
        sensor_pos_z = float(row['sensor_pos_z']) if 'sensor_pos_z' in row.fields else 0.0
        sensor_pos = torch.tensor([sensor_pos_x, sensor_pos_y, sensor_pos_z], dtype=torch.float32) / 1000. # convert to km

        grp_t_np, grp_c_np = reduce_by_window(raw_t, self.grouping_window_ns)

        # ------------------------------------------------ tensors
        rt_original = torch.from_numpy(grp_t_np) # Absolute times
        rc_original = torch.from_numpy(grp_c_np) # Original charges

        # if max_seq_len_padding is exceeded, randomly sample, count-weighted
        if self.max_seq_len_padding is not None and rt_original.numel() > self.max_seq_len_padding:
            # Count-weighted random sampling
            probs = rc_original.float() / rc_original.sum().clamp(min=1e-9) # Avoid division by zero for empty sequences
            if rc_original.sum() <= 1e-9 : # if all charges are zero or empty
                 idx = torch.randperm(rt_original.numel())[:self.max_seq_len_padding] # random sample if no charge
            else:
                idx = torch.multinomial(probs, self.max_seq_len_padding, replacement=False)

            # Sort indices to preserve time order
            idx, _ = torch.sort(idx)
            rt_original = rt_original[idx]
            rc_original = rc_original[idx]

        epsilon = 1e-9 # For log stability
        if rt_original.numel() == 0:
            # Empty sequence case
            return {
                "charges_log_norm": torch.empty(0, dtype=torch.float32),
                "times_log_norm": torch.empty(0, dtype=torch.float32), # Log-normalized absolute times
                "sensor_pos": sensor_pos,
                # The collate_fn will handle creating an attention_mask based on these empty tensors
            }
        
        # Log-normalized charges
        charges_log_norm = torch.log1p(rc_original.clamp(min=0.0))
        
        # Log-normalized ABSOLUTE times
        # Times should be positive before log. rt_original are absolute times, should be >= 0.
        times_log_norm = torch.log(rt_original.clamp(min=0.0) + epsilon)

        return {
            "charges_log_norm": charges_log_norm,
            "times_log_norm": times_log_norm, # These are log-normalized ABSOLUTE times
            "sensor_pos": sensor_pos,
        }
        
def reduce_by_window(arr, window_size):
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    reps, counts = [], []
    i = 0
    while i < n:
        rep = sorted_arr[i]
        j = np.searchsorted(sorted_arr, rep + window_size, side='right')
        reps.append(rep)
        counts.append(j - i)
        i = j
    return np.array(reps), np.array(counts)