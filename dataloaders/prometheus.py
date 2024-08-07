import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import pyarrow.parquet as pq
import glob
from collections import defaultdict

class PrometheusTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if self.cfg['training']:
            train_files = sorted(glob.glob(self.cfg['data_options']['train_data_file'] + '*.parquet'))
            self.train_dataset = PrometheusTimeSeriesDataset(train_files,
                                                             self.cfg['data_options']['num_bins'],
                                                             self.cfg['data_options']['max_time'])
            
        valid_files = sorted(glob.glob(self.cfg['data_options']['valid_data_file'] + '*.parquet'))
        self.valid_dataset = PrometheusTimeSeriesDataset(valid_files,
                                                        self.cfg['data_options']['num_bins'],
                                                        self.cfg['data_options']['max_time'])
            
    def train_dataloader(self):
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
         
class PrometheusTimeSeriesDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        files,
        num_bins,
        max_time):

        self.files = files
        self.num_bins = num_bins
        self.max_time = max_time
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        num_events = np.array(num_events)
        self.cumulative_lengths = np.cumsum(num_events)
        self.dataset_size = self.cumulative_lengths[-1]
        
        self.current_file = ''
        self.current_data = None
        
        self.current_event = []

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        if i < 0 or i >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_lengths, i+1)
        true_idx = i - (self.cumulative_lengths[file_index-1] if file_index > 0 else 0)
                
        if self.current_file != self.files[file_index]:
            self.current_file = self.files[file_index]
            self.current_data = ak.from_parquet(self.files[file_index])
            
        if len(self.current_event) > 0:
            return torch.from_numpy(self.current_event.pop(0)).float()
        else:
            event = self.current_data[true_idx]
            pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                        event.photons.sensor_pos_y.to_numpy(),
                        event.photons.sensor_pos_z.to_numpy(),
                        event.photons.string_id.to_numpy(),
                        event.photons.sensor_id.to_numpy(),
                        event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
            unique_coords_dict = defaultdict(list)
            for i, coord in enumerate(pos_t[:, :5]):
                unique_coords_dict[tuple(coord)].append(i)
            for coord, indices in unique_coords_dict.items():
                mask = np.zeros(pos_t.shape[0], dtype=bool)
                mask[indices] = True
                
                dom_times = pos_t[:,-1][mask]
                first_dom_hit = dom_times.min()
                
                # bin hits on individual sensors
                num_bins = self.num_bins
                max_time = self.max_time
                bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
                
                dom_times = dom_times - first_dom_hit # shift by first hit time
                
                # do not consider hits with time > max_time
                max_time_mask = (dom_times < max_time)
                
                binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
                binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
                
                # put hits with time > max_time in the last bin
                binned_time_counts[-1] += np.sum(~max_time_mask)
                self.current_event.append(binned_time_counts.astype(np.int32))
            return torch.from_numpy(self.current_event.pop(0)).float()

class ParquetFileSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, parquet_file_idxs, batch_size):
       self.data_source = data_source
       self.parquet_file_idxs = parquet_file_idxs
       self.batch_size = batch_size

    def __iter__(self):
        # Determine the number of batches in each parquet file
        num_entries_per_file = [end - start for start, end in zip(self.parquet_file_idxs[:-1], self.parquet_file_idxs[1:])]
      
        # Create an array of file indices, repeated by the number of entries in each file
        file_indices = np.repeat(np.arange(len(num_entries_per_file)), num_entries_per_file)
      
        # Shuffle the file indices
        np.random.shuffle(file_indices)

        for file_index in file_indices:
            start_idx, end_idx = self.parquet_file_idxs[file_index], self.parquet_file_idxs[file_index + 1]
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            
            # Yield batches of indices ensuring all entries are seen
            for i in range(0, len(indices), self.batch_size):
                yield from indices[i:i+self.batch_size].tolist()

    def __len__(self):
       return len(self.data_source)