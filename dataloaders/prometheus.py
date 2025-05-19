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

from .data_utils import get_file_names, InterleavedFileBatchSampler, variable_length_collate_fn, calculate_summary_statistics
from functools import partial # For passing max_seq_len_padding to collate_fn

class PrometheusTimeSeriesDataModule(pl.LightningDataModule): # Renamed from PrometheusDataModule
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
            self.train_dataset = PrometheusTimeSeriesDataset(
                train_files_by_folder,
                train_events_per_file_by_folder,
                self.cfg
            )
            
        valid_files_by_folder, valid_events_per_file_by_folder = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files'] # Typically false for validation
        )
        self.valid_dataset = PrometheusTimeSeriesDataset(
            valid_files_by_folder,
            valid_events_per_file_by_folder,
            self.cfg
        )

    def train_dataloader(self):
        """Returns the training dataloader."""
        batch_sampler = InterleavedFileBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.cfg['training_options']['batch_size'],
            shuffle_intra_batch=True,
            shuffle_files_within_folder=self.cfg['data_options'].get('shuffle_files', True) # Align with global file shuffle option
        )
        collate_fn_with_padding = partial(variable_length_collate_fn)

        dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                            batch_sampler=batch_sampler, # Use batch_sampler
                                            collate_fn=collate_fn_with_padding,
                                            pin_memory=True,
                                            persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader

    def val_dataloader(self):
        """Returns the validation dataloader."""
        batch_sampler = InterleavedFileBatchSampler(
            dataset=self.valid_dataset,
            batch_size=self.cfg['training_options']['batch_size'],
            shuffle_intra_batch=False, # No need to shuffle within validation batches
            shuffle_files_within_folder=False # Keep validation file order consistent
        )
        collate_fn_with_padding = partial(variable_length_collate_fn)

        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_sampler=batch_sampler, # Use batch_sampler
                                           collate_fn=collate_fn_with_padding,
                                           pin_memory=True,
                                           persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        batch_sampler = InterleavedFileBatchSampler(
            dataset=self.valid_dataset, # Using validation dataset for test
            batch_size=self.cfg['training_options']['batch_size'],
            shuffle_intra_batch=False,
            shuffle_files_within_folder=False
        )
        collate_fn_with_padding = partial(variable_length_collate_fn)
        
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_sampler=batch_sampler, # Use batch_sampler
                                           collate_fn=collate_fn_with_padding,
                                           pin_memory=True,
                                           persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                           num_workers=self.cfg['training_options']['num_workers'])

class PrometheusTimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset class for Prometheus data, handling multiple folders.
    Loads data from parquet files and preprocesses it for the model.
    """
    def __init__(self,
                 files_by_folder: List[List[str]],
                 events_per_file_by_folder: List[List[int]],
                 cfg: Dict,
                 cache_size: int = 5): # Added cache_size for Parquet data
        self.files_by_folder = files_by_folder
        self.events_per_file_by_folder = events_per_file_by_folder
        self.cfg = cfg
        self.grouping_window_ns = self.cfg['data_options'].get('grouping_window_ns', 2.0)
        self.max_seq_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None)
        
        self.num_folders = len(self.files_by_folder)
        self.folder_total_events = [sum(counts) for counts in self.events_per_file_by_folder]
        self.dataset_size = sum(self.folder_total_events)

        # Cumulative lengths for files within each folder
        self.cumulative_lengths_by_folder_file = []
        for folder_event_counts in self.events_per_file_by_folder:
            if folder_event_counts: # Check if the list is not empty
                self.cumulative_lengths_by_folder_file.append(
                    np.concatenate(([0], np.cumsum(np.array(folder_event_counts, dtype=np.int64))))
                )
            else: # Handle empty folder (no files or files with 0 events)
                self.cumulative_lengths_by_folder_file.append(np.array([0], dtype=np.int64))

        # Simple FIFO cache for loaded Parquet data (ak.RecordBatch)
        self.cache_size = cache_size
        self.data_cache = OrderedDict() # Stores {file_path: ak_data}
        self.current_file_path_per_folder_file_tuple = {}

    def __len__(self):
        return self.dataset_size # Total events across all folders
    
    def _get_file_path_and_event_index_in_file(self, folder_idx: int, event_idx_in_folder: int) -> Tuple[str, int]:
        """
        Given a folder index and an event index within that folder,
        determines the specific file and the event's index within that file.
        """

        cumulative_lengths_for_folder = self.cumulative_lengths_by_folder_file[folder_idx]

        file_idx_in_folder = int(np.searchsorted(cumulative_lengths_for_folder, event_idx_in_folder + 1) - 1)
        event_idx_in_file = int(event_idx_in_folder - cumulative_lengths_for_folder[file_idx_in_folder])
        
        file_path = self.files_by_folder[folder_idx][file_idx_in_folder]
        return file_path, event_idx_in_file

    def _load_data_from_file(self, file_path: str):
        """Loads data from a parquet file, using a cache."""
        if file_path in self.data_cache:
            # Move to end to mark as recently used
            self.data_cache.move_to_end(file_path)
            return self.data_cache[file_path]
        data = ak.from_parquet(file_path)
        # Add to cache
        if len(self.data_cache) >= self.cache_size:
            self.data_cache.popitem(last=False) # Remove oldest item
        self.data_cache[file_path] = data
        return data

    def __getitem__(self, item_tuple: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Returns one training example.
        item_tuple: (folder_idx, event_idx_in_folder)
        """
        folder_idx, event_idx_in_folder = item_tuple

        file_path, true_idx = self._get_file_path_and_event_index_in_file(folder_idx, event_idx_in_folder)
        
        # Load data using cache
        ak_data_for_file = self._load_data_from_file(file_path)

        row = ak_data_for_file[true_idx]

        # ------------------------------------------------ raw hit times
        raw_t = np.asarray(row['hits_t'], dtype=np.float32)
        
        # Extract sensor position
        sensor_pos_x = float(row['sensor_pos_x']) if 'sensor_pos_x' in row.fields else 0.0
        sensor_pos_y = float(row['sensor_pos_y']) if 'sensor_pos_y' in row.fields else 0.0
        sensor_pos_z = float(row['sensor_pos_z']) if 'sensor_pos_z' in row.fields else 0.0
        sensor_pos = torch.tensor([sensor_pos_x, sensor_pos_y, sensor_pos_z], dtype=torch.float32) / 1000. # convert to km

        grp_t_np, grp_c_np = reduce_by_window(raw_t, self.grouping_window_ns)

        # Calculate summary statistics before potential sampling
        summary_stats_np = calculate_summary_statistics(grp_t_np, grp_c_np)
        epsilon = 1e-6
        summary_stats_np = np.log(summary_stats_np + epsilon) # Log-normalize summary statistics
        summary_stats = torch.from_numpy(summary_stats_np)
        
        # ------------------------------------------------ tensors + log-normalization
        rt = torch.from_numpy(grp_t_np)
        rc = torch.from_numpy(grp_c_np)
        
        if rt.numel():  # non-empty
            nt = torch.log(rt + epsilon)
            nq = torch.log(rc + epsilon)
        else:
            nt = torch.empty(0, dtype=torch.float32)
            nq = torch.empty(0, dtype=torch.float32)
        
        # if max_seq_len_padding is exceeded, randomly sample, count-weighted
        if self.max_seq_len_padding is not None and rt.numel() > self.max_seq_len_padding:
            # Count-weighted random sampling
            probs = rc.float() / rc.sum()
            idx = torch.multinomial(probs, self.max_seq_len_padding, replacement=False)
            # Sort indices to preserve time order
            idx, _ = torch.sort(idx)
            nt = nt[idx]
            nq = nq[idx]
            rt = rt[idx]
            rc = rc[idx]
        
        return {
            "times":           nt,
            "counts":          nq,
            "raw_times":       rt,
            "raw_counts":      rc,
            "sequence_length": torch.tensor(rt.numel(), dtype=torch.long),
            "summary_stats":   summary_stats,
            "sensor_pos":      sensor_pos,
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