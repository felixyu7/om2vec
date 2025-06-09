import os
import glob
import torch
import numpy as np
import math
import pytorch_lightning as pl
import awkward as ak
import pyarrow.parquet as pq
from typing import Dict, Tuple, List
from collections import OrderedDict
from tqdm import tqdm

from .data_utils import get_file_names, variable_length_collate_fn
from functools import partial

class PrometheusTimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for Prometheus dataset.
    Now loads data directly from event-based parquet files.
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
            train_files_by_folder = get_file_names(
                self.cfg['data_options']['train_data_files'],
                self.cfg['data_options']['train_data_file_ranges'],
                self.cfg['data_options']['shuffle_files']
            )
            self.train_dataset = PrometheusTimeSeriesDataset(
                train_files_by_folder,
                self.cfg,
                dataset_type='train'
            )
            
        valid_files_by_folder = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            False # Never shuffle validation files
        )
        self.valid_dataset = PrometheusTimeSeriesDataset(
            valid_files_by_folder,
            self.cfg,
            dataset_type='validation'
        )

    def train_dataloader(self):
        """Returns the training dataloader."""
        collate_fn_with_padding = partial(variable_length_collate_fn)

        dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size=self.cfg['training_options']['batch_size'],
                                            shuffle=True,
                                            collate_fn=collate_fn_with_padding,
                                            pin_memory=True,
                                            persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader

    def val_dataloader(self):
        """Returns the validation dataloader."""
        collate_fn_with_padding = partial(variable_length_collate_fn)

        return torch.utils.data.DataLoader(self.valid_dataset,
                                           batch_size=self.cfg['training_options']['batch_size'],
                                           shuffle=False,
                                           collate_fn=collate_fn_with_padding,
                                           pin_memory=True,
                                           persistent_workers=True if self.cfg['training_options']['num_workers'] > 0 else False,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        return self.val_dataloader()

class PrometheusTimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset class for Prometheus data.
    This dataset reads event-based parquet files, builds an index of all sensors,
    and returns data for one sensor at a time.
    """
    def __init__(self,
                 files_by_folder: List[List[str]],
                 cfg: Dict,
                 dataset_type: str = 'train',
                 cache_size: int = 10):
        self.files_by_folder = files_by_folder
        self.cfg = cfg
        self.grouping_window_ns = self.cfg['data_options'].get('grouping_window_ns', 2.0)
        self.max_seq_len_padding = self.cfg['data_options'].get('max_seq_len_padding', None)
        
        # Simple FIFO cache for loaded Parquet data (ak.RecordBatch)
        self.cache_size = cache_size
        self.data_cache = OrderedDict()

        # Build an index that maps a dataset index to a specific sensor in a specific event in a specific file
        self.sensor_index = self._build_sensor_index()
        self.dataset_size = len(self.sensor_index)
        print(f"[{dataset_type}] Created dataset with {self.dataset_size} total sensors.")

    def _build_sensor_index(self) -> List[Tuple[str, int, tuple]]:
        """
        Scans all parquet files to build an index of (file_path, event_idx_in_file, sensor_key).
        A sensor_key is a tuple of (string_id, sensor_id).
        """
        index = []
        print("Building sensor index...")
        # Flatten the list of files from all folders
        all_files = [file_path for folder in self.files_by_folder for file_path in folder]

        for file_path in tqdm(all_files):
            try:
                pf = pq.ParquetFile(file_path)
                # Here we assume files are small enough to be loaded into memory for index building.
                # For very large files, a row-group-based approach would be needed.
                data = ak.from_parquet(file_path, columns=["photons.string_id", "photons.sensor_id"])
                
                for event_idx, event in enumerate(data):
                    photons = event.photons
                    if photons is None or len(photons.string_id) == 0:
                        continue
                    
                    # Create a unique sensor key tuple for each hit
                    sensor_keys = ak.zip([photons.string_id, photons.sensor_id])
                    
                    # Find the unique sensor keys in this event
                    unique_sensor_keys = ak.unique(sensor_keys)
                    
                    for key in unique_sensor_keys:
                        # key is a record like {'0': string_id, '1': sensor_id}
                        sensor_tuple = (key['0'], key['1'])
                        index.append((file_path, event_idx, sensor_tuple))
            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {e}. Skipping.")
                continue
        return index

    def __len__(self):
        return self.dataset_size

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one training example for a single sensor.
        """
        file_path, event_idx, sensor_key = self.sensor_index[idx]
        
        ak_data_for_file = self._load_data_from_file(file_path)
        
        event = ak_data_for_file[event_idx]
        photons = event.photons
        
        string_id_to_match, sensor_id_to_match = sensor_key
        
        # Create a mask to select hits for the specific sensor
        mask = (photons.string_id == string_id_to_match) & (photons.sensor_id == sensor_id_to_match)
        
        # Get times and positions for the selected sensor
        sensor_hits_t = photons.t[mask]
        
        # All hits for a sensor have the same position, so we can take the first one
        sensor_pos_x = photons.sensor_pos_x[mask][0]
        sensor_pos_y = photons.sensor_pos_y[mask][0]
        sensor_pos_z = photons.sensor_pos_z[mask][0]
        
        sensor_pos = torch.tensor([sensor_pos_x, sensor_pos_y, sensor_pos_z], dtype=torch.float32) / 1000.0 # convert to km

        raw_t = np.asarray(sensor_hits_t, dtype=np.float32)
        
        grp_t_np, grp_c_np = reduce_by_window(raw_t, self.grouping_window_ns)

        rt_original = torch.from_numpy(grp_t_np)
        rc_original = torch.from_numpy(grp_c_np)

        if self.max_seq_len_padding is not None and rt_original.numel() > self.max_seq_len_padding:
            probs = rc_original.float() / rc_original.sum().clamp(min=1e-9)
            if rc_original.sum() <= 1e-9:
                idx_perm = torch.randperm(rt_original.numel())[:self.max_seq_len_padding]
            else:
                idx_perm = torch.multinomial(probs, self.max_seq_len_padding, replacement=False)
            
            idx_sorted, _ = torch.sort(idx_perm)
            rt_original = rt_original[idx_sorted]
            rc_original = rc_original[idx_sorted]

        epsilon = 1e-9
        if rt_original.numel() == 0:
            return {
                "charges_log_norm": torch.empty(0, dtype=torch.float32),
                "times_log_norm": torch.empty(0, dtype=torch.float32),
                "sensor_pos": sensor_pos,
            }
        
        charges_log_norm = torch.log1p(rc_original.clamp(min=0.0))
        times_log_norm = torch.log(rt_original.clamp(min=0.0) + epsilon)

        return {
            "charges_log_norm": charges_log_norm,
            "times_log_norm": times_log_norm,
            "sensor_pos": sensor_pos,
        }

def reduce_by_window(arr, window_size):
    if arr.size == 0:
        return np.array([]), np.array([])
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
    return np.array(reps, dtype=np.float32), np.array(counts, dtype=np.float32)