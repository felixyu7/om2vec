import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
import awkward as ak
import pyarrow.parquet as pq

from .data_utils import get_file_names, ParquetFileSampler, variable_length_collate_fn # Import new collate_fn
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
            train_files = get_file_names(
                self.cfg['data_options']['train_data_files'], 
                self.cfg['data_options']['train_data_file_ranges'],
                self.cfg['data_options']['shuffle_files']
            )
            self.train_dataset = PrometheusTimeSeriesDataset(train_files, self.cfg) # Pass cfg, Renamed
            
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files']
        )
        self.valid_dataset = PrometheusTimeSeriesDataset(valid_files, self.cfg) # Pass cfg, Renamed

    def train_dataloader(self):
        """Returns the training dataloader."""
        # Using standard shuffle and new collate_fn
        # sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        
        # Get max_seq_len_padding from config for the collate function
        max_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None)
        collate_fn_with_padding = partial(variable_length_collate_fn, max_seq_len_padding=max_len_padding)

        dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size = self.cfg['training_options']['batch_size'],
                                            shuffle=True, # Use standard shuffle
                                            collate_fn=collate_fn_with_padding, # Use new collate_fn
                                            pin_memory=True,
                                            persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader

    def val_dataloader(self):
        """Returns the validation dataloader."""
        # sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        
        max_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None)
        collate_fn_with_padding = partial(variable_length_collate_fn, max_seq_len_padding=max_len_padding)

        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size = self.cfg['training_options']['batch_size'],
                                           shuffle=False, # No shuffle for validation
                                           collate_fn=collate_fn_with_padding, # Use new collate_fn
                                           pin_memory=True,
                                           persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        # sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        
        max_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None)
        collate_fn_with_padding = partial(variable_length_collate_fn, max_seq_len_padding=max_len_padding)
        
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size = self.cfg['training_options']['batch_size'],
                                           shuffle=False, # No shuffle for test
                                           collate_fn=collate_fn_with_padding, # Use new collate_fn
                                           pin_memory=True,
                                           persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                           num_workers=self.cfg['training_options']['num_workers'])

class PrometheusTimeSeriesDataset(torch.utils.data.Dataset): # Renamed from PrometheusDataset
    """
    Dataset class for Prometheus data.
    
    Handles loading data from parquet files and preprocessing it for the model.
    """
    def __init__(self, files, cfg): # Added cfg
        self.files = files
        self.cfg = cfg # Store cfg
        # self.max_seq_len is no longer used here, padding handled by collate_fn
        self.grouping_window_ns = self.cfg['data_options'].get('grouping_window_ns', 2.0) # Get from cfg
        
        # Count number of events in each file
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        num_events = np.array(num_events)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(num_events)))
        self.dataset_size = self.cumulative_lengths[-1]
        
        self.current_file = ''
        self.current_data = None
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, i: int):
        """
        Return one training example.

        Photon hits are grouped into `grouping_window_ns` (e.g., 2ns) windows.
        Each group becomes an event with (time_of_first_hit_in_group, num_hits_in_group).
        Outputs variable-length sequences. Padding is handled by collate_fn.
        """
        # ---------- index bookkeeping ----------
        if i < 0 or i >= self.dataset_size:
            raise IndexError("Index out of range")

        file_index = np.searchsorted(self.cumulative_lengths, i + 1) - 1
        true_idx   = i - self.cumulative_lengths[file_index]

        # ---------- lazy-load current parquet ----------
        if self.current_file != self.files[file_index]:
            self.current_file = self.files[file_index]
            self.current_data = ak.from_parquet(self.current_file)

        row = self.current_data[true_idx]

        # ---------- raw per-hit arrays ----------
        time_column = self.cfg['data_options'].get('time_column', 't')
        raw_t = np.asarray(row[time_column], dtype=np.float32)

        if len(raw_t) == 0: # Handle empty sequences
            grouped_times_np = np.array([], dtype=np.float32)
            grouped_counts_np = np.array([], dtype=np.float32)
        else:
            # Sort times for grouping
            sorted_indices = np.argsort(raw_t)
            raw_t_sorted = raw_t[sorted_indices]

            grouped_times = []
            grouped_counts = []
            
            current_window_start_time = raw_t_sorted[0]
            current_window_count = 0

            for t_hit in raw_t_sorted:
                if t_hit < current_window_start_time + self.grouping_window_ns:
                    current_window_count += 1
                else:
                    # Finalize previous window
                    grouped_times.append(current_window_start_time)
                    grouped_counts.append(current_window_count)
                    # Start new window
                    current_window_start_time = t_hit
                    current_window_count = 1
            
            # Add the last window
            if current_window_count > 0:
                grouped_times.append(current_window_start_time)
                grouped_counts.append(current_window_count)

            grouped_times_np = np.array(grouped_times, dtype=np.float32)
            grouped_counts_np = np.array(grouped_counts, dtype=np.float32)

        # ---------- Convert to Tensors and normalise ----------
        raw_grouped_times_tensor = torch.from_numpy(grouped_times_np).float()
        raw_grouped_counts_tensor = torch.from_numpy(grouped_counts_np).float()

        # Apply log_transform (ensure log_transform handles empty tensors or zero values appropriately)
        # For empty sequences, log_transform might need adjustment or these will be empty tensors.
        if raw_grouped_times_tensor.numel() > 0:
            norm_t = torch.log(raw_grouped_times_tensor + 1)
            norm_q = torch.log(raw_grouped_counts_tensor + 1)
        else:
            norm_t = torch.empty(0, dtype=torch.float32)
            norm_q = torch.empty(0, dtype=torch.float32)

        return {
            "times": norm_t,  # Log-transformed absolute times of grouped events
            "counts": norm_q, # Log-transformed counts of grouped events
            "raw_times": raw_grouped_times_tensor, # Non-transformed, for calculating inter-event times later
            "raw_counts": raw_grouped_counts_tensor, # Non-transformed counts
            "sequence_length": torch.tensor(len(grouped_times_np), dtype=torch.long)
        }