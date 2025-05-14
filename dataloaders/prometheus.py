import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
import awkward as ak
import pyarrow.parquet as pq

from utils.utils import log_transform # Updated import
from dataloaders.data_utils import get_file_names, ParquetFileSampler

class PrometheusDataModule(pl.LightningDataModule):
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
            self.train_dataset = PrometheusDataset(train_files, self.cfg) # Pass cfg
            
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files']
        )
        self.valid_dataset = PrometheusDataset(valid_files, self.cfg) # Pass cfg

    def train_dataloader(self):
        """Returns the training dataloader."""
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        """Returns the validation dataloader."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           batch_size = self.cfg['training_options']['batch_size'], 
                                           sampler=sampler,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           batch_size = self.cfg['training_options']['batch_size'], 
                                           sampler=sampler,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           num_workers=self.cfg['training_options']['num_workers'])
        
class PrometheusDataset(torch.utils.data.Dataset):
    """
    Dataset class for Prometheus data.
    
    Handles loading data from parquet files and preprocessing it for the model.
    """
    def __init__(self, files, cfg): # Added cfg
        self.files = files
        self.cfg = cfg # Store cfg
        self.max_seq_len = self.cfg['data_options']['max_seq_len']
        
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

        The photon sequence is *never* truncated: if it is longer than
        self.max_seq_len the hits are binned down to that length by time.
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
        time_column = self.cfg['data_options'].get('time_column', 't') # Get time column from config
        raw_t = np.asarray(row[time_column], dtype=np.float32)
        raw_q = np.ones_like(raw_t, dtype=np.float32)      # unit charge per hit, becomes count after binning

        # Prometheus-specific features (sensor_pos, z_summary) are removed for om2vec

        max_len = self.max_seq_len
        cur_len = len(raw_t)

        # ---------- BIN if sequence is too long ----------
        if cur_len > max_len:
            # build max_len equal-width bins between the first and last hit
            t_min, t_max = raw_t.min(), raw_t.max()
            bin_edges = np.linspace(t_min, t_max, max_len + 1, dtype=np.float32)

            # which bin does every hit belong to?
            # np.searchsorted gives 1…max_len; subtract 1 → 0…max_len-1
            bin_idx = np.searchsorted(bin_edges, raw_t, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, max_len - 1)

            # aggregate charge (counts) with bincount
            binned_q = np.bincount(
                bin_idx, weights=raw_q, minlength=max_len
            ).astype(np.float32)

            # aggregate mean time per bin:
            #   mean =   Σ t  /  n  ;  keep n so empty bins → 0
            sums_t   = np.bincount(bin_idx, weights=raw_t, minlength=max_len)
            counts   = np.bincount(bin_idx, minlength=max_len)
            # avoid /0 by using where(counts>0)
            binned_t = np.zeros(max_len, dtype=np.float32)
            non_empty = counts > 0
            binned_t[non_empty] = (sums_t[non_empty] / counts[non_empty]).astype(np.float32)

            # the binned sequence is now the working sequence
            work_t = torch.from_numpy(binned_t)
            work_q = torch.from_numpy(binned_q)
            attention_mask = torch.ones(max_len, dtype=torch.bool)
            active_entries_count = max_len
        else:
            # ---------- original (≤ max_len) path ----------
            work_t = torch.from_numpy(raw_t)
            work_q = torch.from_numpy(raw_q)
            attention_mask = torch.ones(max_len, dtype=torch.bool)
            pad = max_len - cur_len
            if pad:                             # zero-pad to max_len
                work_t = torch.cat([work_t, torch.zeros(pad)], 0)
                work_q = torch.cat([work_q, torch.zeros(pad)], 0)
                attention_mask[cur_len:] = False
            active_entries_count = cur_len

        # ---------- normalise ----------
        norm_t = log_transform(work_t.float()) # Use log_transform
        norm_q = log_transform(work_q.float()) # Use log_transform
        
        # norm_sensor_pos and norm_z_summary removed

        input_tq_sequence = torch.stack([norm_t, norm_q], dim=1)  # (max_len, 2)

        return {
            "input_tq_sequence":   input_tq_sequence,      # (L,2)
            "attention_mask":      attention_mask,         # (L,)
            "active_entries_count": torch.tensor(active_entries_count, dtype=torch.float32) # Crucial for z[0]
        }
        