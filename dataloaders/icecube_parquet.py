import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
import awkward as ak
import pyarrow.parquet as pq
import random
from collections import defaultdict

from .data_utils import get_file_names, IrregularDataCollator

class IcecubeParquetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for IceCube parquet datasets.
    Adapted specifically for Simformer models.
    """
    def __init__(self, cfg): # Removed field, can be in cfg if needed
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
        # Get cache size from config or use default
        
        if self.cfg['training'] or stage == 'fit':
            train_files = get_file_names(
                self.cfg['data_options']['train_data_files'],
                self.cfg['data_options']['train_data_file_ranges'],
                self.cfg['data_options'].get('shuffle_files', False)
            )
            # Pass the full cfg to the dataset
            self.train_dataset = IceCube_Parquet_Dataset(files=train_files, cfg=self.cfg)
            
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options'].get('shuffle_files', False)
        )
        
        # Pass the full cfg to the dataset
        self.valid_dataset = IceCube_Parquet_Dataset(files=valid_files, cfg=self.cfg)
            
    def train_dataloader(self):
        """Returns the training dataloader."""
        training_options = self.cfg.get('training_options', {})
        batch_size = training_options.get('batch_size', 32)
        num_workers = training_options.get('num_workers', 0)
        
        # Use BatchedCacheParquetSampler for full randomization across cache
        sampler = BatchedCacheParquetSampler(self.train_dataset, batch_size, shuffle=True)
        collate_fn = IrregularDataCollator()
        
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            num_workers=num_workers
        )
    
    def val_dataloader(self):
        """Returns the validation dataloader."""
        training_options = self.cfg.get('training_options', {})
        batch_size = training_options.get('batch_size', 32)
        num_workers = training_options.get('num_workers', 0)
        
        sampler = BatchedCacheParquetSampler(self.valid_dataset, batch_size, shuffle=False)
        collate_fn = IrregularDataCollator()
        
        return torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            num_workers=num_workers
        )

    def test_dataloader(self):
        """Returns the test dataloader."""
        return self.val_dataloader()
        
class BatchedCacheParquetSampler(torch.utils.data.Sampler):
    """
    Sampler that loads files in batches and completely randomizes events across those files.
    When events from one batch of files are exhausted, it loads the next batch.
    This ensures complete randomization while maintaining efficient caching.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        file_indices = list(range(len(self.dataset.files)))
        if self.shuffle:
            random.shuffle(file_indices)

        cache_size = self.dataset.cache_size
        for i in range(0, len(file_indices), cache_size):
            cached_files = file_indices[i:i+cache_size]
            # Build a pool of (file_idx, local_event_idx) tuples
            pools = {f: list(range(
                        self.dataset.cumulative_lengths[f],
                        self.dataset.cumulative_lengths[f+1])) for f in cached_files}

            while pools:                          # until every cached file is exhausted
                f = random.choice(list(pools))    # pick a random file in the cache
                yield pools[f].pop()              # yield one event from it
                if not pools[f]:                  # file empty → remove
                    pools.pop(f)
        
    def __len__(self):
        return len(self.dataset)

class IceCube_Parquet_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for IceCube parquet data.
    
    Handles loading data from parquet files and preprocessing it for the model.
    With efficient multi-file caching capability.
    """
    def __init__(self, files, cfg):
        self.files = files
        self.cfg = cfg
        self.data_options = cfg['data_options']
        
        self.cache_size = min(self.data_options.get('cache_size', 20), len(files))
        self.max_oms_per_event = self.data_options.get('max_oms_per_event', 50)
        self.max_photons_per_om = self.data_options.get('max_photons_per_om', 128)

        # For om2vec, we always need pulse series.
        # The flags use_pulse_series and use_latent_representation are less relevant here.
        # We will directly process pulses.

        # Count number of events in each file
        self.num_events_per_file = []
        for file in self.files:
            data = pq.ParquetFile(file)
            self.num_events_per_file.append(data.metadata.num_rows)
        self.num_events_per_file = np.array(self.num_events_per_file)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(self.num_events_per_file)))
        self.dataset_size = self.cumulative_lengths[-1]
        
        # Simple dict-based cache (much faster than LRU tracking for each access)
        self.file_cache = {}
        
    def __len__(self):
        return self.dataset_size
    
    def _get_event(self, file_index, event_index):
        # Check if file is already loaded in cache
        if file_index not in self.file_cache:
            # If cache is full, evict only one file (the first key)
            if len(self.file_cache) >= self.cache_size:
                key_to_remove = next(iter(self.file_cache))
                del self.file_cache[key_to_remove]
            # Load the file into cache
            self.file_cache[file_index] = ak.from_parquet(self.files[file_index])
        return self.file_cache[file_index][event_index]
    
    def __getitem__(self, i):
        """
        Get an item from the dataset, representing one event structured for om2vec.

        Args:
            i: Index of the item (event).

        Returns:
            A dictionary containing:
            'all_om_hits': (max_oms_per_event, max_photons_per_om, 2) tensor for (t,q)
            'all_om_sensor_pos': (max_oms_per_event, 3) tensor for sensor [x,y,z]
            'om_mask': (max_oms_per_event,) boolean tensor, True for valid OMs
            'hit_mask': (max_oms_per_event, max_photons_per_om) boolean tensor, True for valid hits
            'label': (4,) tensor for event-level label [norm_energy, dir_x, dir_y, dir_z]
            'event_id': scalar, original index i, for tracking/debugging
        """
        if i < 0 or i >= self.dataset_size:
            raise IndexError("Index out of range")
        
        file_index = np.searchsorted(self.cumulative_lengths, i + 1) - 1
        true_idx = i - self.cumulative_lengths[file_index]
        
        event = self._get_event(file_index, true_idx)

        # Initialize output tensors
        all_om_hits = torch.zeros((self.max_oms_per_event, self.max_photons_per_om, 2), dtype=torch.float32)
        all_om_sensor_pos = torch.zeros((self.max_oms_per_event, 3), dtype=torch.float32)
        om_mask = torch.zeros((self.max_oms_per_event,), dtype=torch.bool)
        hit_mask = torch.zeros((self.max_oms_per_event, self.max_photons_per_om), dtype=torch.bool)

        # Extract MC truth information (event-level label)
        direction = event.mc_truth.primary_direction.to_numpy() # Should be (3,)
        energy = np.log10(event.mc_truth.primary_energy.to_numpy()) # Scalar
        
        energy_min = self.data_options.get('energy_norm_min', 2.0) # Default log10(100 GeV)
        energy_max = self.data_options.get('energy_norm_max', 8.0) # Default log10(100 PeV)
        energy_norm = 2 * (energy - energy_min) / (energy_max - energy_min) - 1
        
        label_list = [energy_norm] + direction.tolist()
        label_tensor = torch.tensor(label_list, dtype=torch.float32)

        # Process pulses for each OM
        # event.pulses.pulse_times is like [[t1,t2..]_om1, [tA,tB..]_om2, ...]
        # event.pulses.sensor_pos_x is like [x_om1, x_om2, ...]
        
        num_oms_in_event = len(event.pulses.sensor_pos_x)
        num_oms_to_process = min(num_oms_in_event, self.max_oms_per_event)

        om_indices = list(range(num_oms_in_event))
        if self.data_options.get('shuffle_oms_within_event', False) and num_oms_in_event > 0 : # Optional: shuffle OMs if desired
            random.shuffle(om_indices)
        
        processed_om_count = 0
        for om_idx_in_event_list in om_indices:
            if processed_om_count >= num_oms_to_process:
                break

            om_pulse_times = np.array(event.pulses.pulse_times[om_idx_in_event_list])
            om_pulse_charges = np.array(event.pulses.pulse_charges[om_idx_in_event_list])

            if len(om_pulse_times) == 0: # Skip OMs with no pulses
                continue

            # Sort pulses by time for consistent processing (truncation, summary stats)
            sort_indices = np.argsort(om_pulse_times)
            om_pulse_times_sorted = om_pulse_times[sort_indices]
            om_pulse_charges_sorted = om_pulse_charges[sort_indices]

            om_sensor_pos = torch.tensor([
                event.pulses.sensor_pos_x[om_idx_in_event_list],
                event.pulses.sensor_pos_y[om_idx_in_event_list],
                event.pulses.sensor_pos_z[om_idx_in_event_list]
            ], dtype=torch.float32)

            # Store sensor position
            all_om_sensor_pos[processed_om_count] = om_sensor_pos
            om_mask[processed_om_count] = True # Mark this OM as valid

            # Pad/truncate (t,q) sequence for this OM
            num_hits_in_om = len(om_pulse_times_sorted)
            num_hits_to_take = min(num_hits_in_om, self.max_photons_per_om)
            
            # Store (t,q) pairs
            # Normalization will be handled by the model or om_processing utils
            # to allow flexibility (e.g. relative times calculated after this step)
            all_om_hits[processed_om_count, :num_hits_to_take, 0] = torch.from_numpy(om_pulse_times_sorted[:num_hits_to_take])
            all_om_hits[processed_om_count, :num_hits_to_take, 1] = torch.from_numpy(om_pulse_charges_sorted[:num_hits_to_take])
            hit_mask[processed_om_count, :num_hits_to_take] = True # Mark valid hits

            processed_om_count += 1
            
        return {
            "all_om_hits": all_om_hits,
            "all_om_sensor_pos": all_om_sensor_pos,
            "om_mask": om_mask,
            "hit_mask": hit_mask,
            "label": label_tensor,
            "event_id": torch.tensor(i, dtype=torch.long) # For tracking
        }