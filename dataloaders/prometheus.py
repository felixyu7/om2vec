import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
import awkward as ak
import pyarrow.parquet as pq
from typing import Dict

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
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        collate_fn_with_padding = partial(variable_length_collate_fn)

        dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size = self.cfg['training_options']['batch_size'],
                                            sampler = sampler,
                                            collate_fn=collate_fn_with_padding, # Use new collate_fn
                                            pin_memory=True,
                                            persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader

    def val_dataloader(self):
        """Returns the validation dataloader."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        collate_fn_with_padding = partial(variable_length_collate_fn)

        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size = self.cfg['training_options']['batch_size'],
                                           sampler=sampler,
                                           collate_fn=collate_fn_with_padding, # Use new collate_fn
                                           pin_memory=True,
                                           persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        collate_fn_with_padding = partial(variable_length_collate_fn)
        
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size = self.cfg['training_options']['batch_size'],
                                           sampler=sampler,
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
        self.max_seq_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None) # Get from cfg
        
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
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Returns one training example with *both* fixed-window grouping and
        adaptive length-capping already applied.
        """

        # -------------------------------------------------- index bookkeeping
        if i < 0 or i >= self.dataset_size:
            raise IndexError("Index out of range")

        file_idx = int(np.searchsorted(self.cumulative_lengths, i + 1) - 1)
        true_idx = int(i - self.cumulative_lengths[file_idx])

        # ------------------------------------------------ parquet row (lazy)
        if self.current_file != self.files[file_idx]:
            self.current_file = self.files[file_idx]
            self.current_data = ak.from_parquet(self.current_file)

        row = self.current_data[true_idx]

        # ------------------------------------------------ raw hit times
        raw_t = np.asarray(row['hits_t'], dtype=np.float32)

        # ------------------------------------------------ 1) fixed window grouping
        if raw_t.size == 0:                                  # empty fast-path
            grp_t_np = np.empty(0, np.float32)
            grp_c_np = np.empty(0, np.float32)
        else:
            t_sorted     = np.sort(raw_t)
            gaps         = np.diff(t_sorted) >= self.grouping_window_ns
            split_pts    = np.nonzero(gaps)[0] + 1
            chunks       = np.split(t_sorted, split_pts)

            grp_t_np = np.fromiter((c[0]   for c in chunks),
                                count=len(chunks), dtype=np.float32)
            grp_c_np = np.fromiter((c.size for c in chunks),
                                count=len(chunks), dtype=np.float32)

        # ------------------------------------------------ 2) adaptive merge-down
        if (self.max_seq_len_padding is not None and
            grp_t_np.size > self.max_seq_len_padding):

            # keep everything on torch to avoid host/device swaps
            rt = torch.from_numpy(grp_t_np)
            rc = torch.from_numpy(grp_c_np)

            while rt.size(0) > self.max_seq_len_padding:
                idx = torch.argmin(torch.diff(rt))           # nearest pair
                rc[idx] += rc[idx + 1]                       # merge counts
                rt = torch.cat((rt[:idx + 1], rt[idx + 2:]))
                rc = torch.cat((rc[:idx + 1], rc[idx + 2:]))

            grp_t_np = rt.numpy(force=True)                  # still float32
            grp_c_np = rc.numpy(force=True)

        # ------------------------------------------------ tensors + log space
        rt = torch.from_numpy(grp_t_np)
        rc = torch.from_numpy(grp_c_np)

        if rt.numel():                                       # non-empty
            nt = torch.log(rt + 1)
            nq = torch.log(rc + 1)
        else:
            nt = torch.empty(0, dtype=torch.float32)
            nq = torch.empty(0, dtype=torch.float32)

        return {
            "times":           nt,
            "counts":          nq,
            "raw_times":       rt,
            "raw_counts":      rc,
            "sequence_length": torch.tensor(rt.numel(), dtype=torch.long),
        }