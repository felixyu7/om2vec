import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import glob
import os
import random
from torch.utils.data import Sampler, Dataset
from typing import List, Dict, Tuple
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm

def get_file_names(data_dirs: List[str], ranges: List[List[int]], shuffle_files: bool = False) -> List[List[str]]:
    """
    Get file names from directories within specified ranges, grouped by directory.
    """
    files_by_folder = []

    for i, directory in enumerate(data_dirs):
        all_files_in_dir = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        
        file_range = ranges[i]
        selected_files_in_dir = all_files_in_dir[file_range[0]:file_range[1]]

        if not selected_files_in_dir:
            files_by_folder.append([])
            continue

        if shuffle_files:
            random.shuffle(selected_files_in_dir)
        
        files_by_folder.append(selected_files_in_dir)
            
    return files_by_folder

def variable_length_collate_fn(batch: List[List[Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """
    Collate for event-centric data with variable batch sizes.
    Each item in the batch is a list of sensors for one event.
    This function flattens all sensors from all events in the batch into single tensors
    and creates an 'event_indices' tensor to map each sensor back to its event.
    """
    # batch is a list of events, e.g., [[event1_sensor1, ...], [event2_sensor1, ...]]
    events = [e for e in batch if e] # Filter out empty events
    if not events:
        return {}

    # Create a flat list of all sensors and a corresponding event index for each sensor
    all_sensors = []
    event_indices = []
    for i, event_sensors in enumerate(events):
        all_sensors.extend(event_sensors)
        event_indices.extend([i] * len(event_sensors))

    if not all_sensors:
        return {}

    bsz = len(all_sensors)
    device = all_sensors[0]["charges_log_norm"].device
    dtype = all_sensors[0]["charges_log_norm"].dtype

    # Pad all sensor sequences to the max length in the batch
    seq_lens_list = [item["charges_log_norm"].numel() for item in all_sensors]
    max_len = max(seq_lens_list) if seq_lens_list else 0

    charges_padded = torch.zeros(bsz, max_len, dtype=dtype, device=device)
    times_padded = torch.zeros(bsz, max_len, dtype=dtype, device=device)
    attention_mask = torch.ones(bsz, max_len, dtype=torch.bool, device=device)
    sensor_pos_batched = torch.stack([item["sensor_pos"] for item in all_sensors])

    for i, item in enumerate(all_sensors):
        L = item["charges_log_norm"].numel()
        if L > 0:
            charges_padded[i, :L] = item["charges_log_norm"]
            times_padded[i, :L] = item["times_log_norm"]
            attention_mask[i, :L] = False

    # Collect truth values for each event (they are the same for all sensors in an event)
    truth_zenith = torch.stack([event[0]["truth_zenith"] for event in events])
    truth_azimuth = torch.stack([event[0]["truth_azimuth"] for event in events])
    truth_energy = torch.stack([event[0]["truth_energy"] for event in events])
    
    event_indices_tensor = torch.tensor(event_indices, dtype=torch.long, device=device)

    return {
        "charges_log_norm_padded": charges_padded,
        "times_log_norm_padded": times_padded,
        "attention_mask": attention_mask,
        "sensor_pos_batched": sensor_pos_batched,
        "truth_zenith": truth_zenith,
        "truth_azimuth": truth_azimuth,
        "truth_energy": truth_energy,
        "event_indices": event_indices_tensor,
    }

class FileAwareSampler(Sampler):
    """
    A sampler that improves data locality for datasets indexed by sensor.
    It groups sensor indices by their source file, shuffles the files,
    and then yields all sensor indices from one file before moving to the next.
    This minimizes disk I/O by ensuring a file is read only once per epoch.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        
        # Group indices by file_path
        self.indices_by_file = defaultdict(list)
        print("Grouping event indices by file for FileAwareSampler...")
        # Assuming data_source has an 'event_index' attribute
        for i, (file_path, _) in enumerate(tqdm(self.data_source.event_index)):
            self.indices_by_file[file_path].append(i)
        
        self.files = list(self.indices_by_file.keys())

    def __iter__(self):
        # Shuffle the order of files to process
        random.shuffle(self.files)
        
        # Iterate through each file
        for file_path in self.files:
            # Get all sensor indices for this file and shuffle them
            indices_in_file = self.indices_by_file[file_path]
            random.shuffle(indices_in_file)
            
            # Yield all indices from this file
            for index in indices_in_file:
                yield index

    def __len__(self):
        return len(self.data_source)