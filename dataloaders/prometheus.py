import torch
import numpy as np
import lightning.pytorch as pl
import gc, os, glob
from collections import OrderedDict

import awkward as ak
from bisect import bisect_right

from dataloaders.data_utils import get_file_names, RandomChunkSampler

class PrometheusTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if self.cfg['training']:
            train_files = get_file_names(self.cfg['data_options']['train_data_files'], self.cfg['data_options']['train_data_file_ranges'])
            self.train_dataset = PrometheusTimeSeriesDataset(cache_size=32,
                                           reduce_size=-1,
                                           resource_path='/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/GNN/software/src/nunet/resources',
                                           paths=train_files)
            
        valid_files = get_file_names(self.cfg['data_options']['valid_data_files'], self.cfg['data_options']['valid_data_file_ranges'])
        self.valid_dataset = PrometheusTimeSeriesDataset(cache_size=32,
                                        reduce_size=-1,
                                        resource_path='/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/GNN/software/src/nunet/resources',
                                        paths=valid_files)
        
    def train_dataloader(self):
        sampler = RandomChunkSampler(self.train_dataset, batch_size=self.cfg['training_options']['batch_size'], chunks=self.train_dataset.chunks)
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers']-1,
                                            prefetch_factor=2)
        return dataloader
    
    def val_dataloader(self):
        sampler = RandomChunkSampler(self.valid_dataset, batch_size=self.cfg['training_options']['batch_size'], chunks=self.valid_dataset.chunks)
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers']-1,
                                            prefetch_factor=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers']-1,
                                            prefetch_factor=2)

class PrometheusTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_size=32,
        reduce_size=-1,
        resource_path='',
        paths=[]
    ):  
        self.reduce_size, self.cache_size = reduce_size, cache_size
        self.resource_path = resource_path

        self.path_dict = {}  # keep track of which folder the files are in
        self.meta_path_dict = {}  # keep track of which folder the meta files are in
        
        self.chunks = []
        self.files = []
        self.meta_files = []
        self.paths = paths
        
        for path in self.paths:
            # Get the number of events in each folder
            base_path = '/' + os.path.join(*path.split('/')[:-1])
            
            # Get the file names and meta file names
            _files = [p.split('/')[-1] for p in sorted(glob.glob(path)) if 'meta' not in p]
            self.files += _files
            for f in _files:
                self.path_dict[f] = base_path
            
            # TODO: eliminate the meta_path_dict
            _meta_files = [p.split('/')[-1] for p in sorted(glob.glob(path + '/*.meta.parquet'))]
            self.meta_files += _meta_files
            for mf in _meta_files:
                self.meta_path_dict[mf] = base_path
        
        file_size = 250000
        self.chunks = np.array([file_size] * len(self.files))
        self.chunk_cumsum = np.cumsum(self.chunks)
        self.cache, self.meta_cache = None, None
        
        # randomly permute files list
        self.files = np.random.permutation(self.files)
        gc.collect()

    def __len__(self):
        return (
            self.chunk_cumsum[-1]
            if self.reduce_size < 0
            else int(self.reduce_size * self.chunk_cumsum[-1])
        )

    def load_data(self, fname):
        
        if self.cache is None:
            self.cache = OrderedDict()
        if self.meta_cache is None:
            self.meta_cache = OrderedDict()
        
        if fname not in self.cache:
            df = ak.from_parquet(os.path.join(self.path_dict[fname], fname)).binned_time_counts.to_numpy().astype(np.float32)
            # df = polars.read_parquet(os.path.join(self.path_dict[fname], fname), low_memory=True)['binned_time_counts'].to_numpy()
            # df = np.vstack(df).astype(np.float32)
            self.cache[fname] = df
            if len(self.cache) > self.cache_size:
                del self.cache[list(self.cache.keys())[0]]

    def __getitem__(self, idx0):
        fidx = bisect_right(self.chunk_cumsum, idx0)
        fname = self.files[fidx]
        idx = int(idx0 - self.chunk_cumsum[fidx - 1]) if fidx > 0 else idx0
        
        self.load_data(fname)
        time_series = self.cache[fname][idx]
        # time_series = data.binned_time_counts.to_numpy()
        time_series = np.log(time_series + 1)

        return torch.from_numpy(time_series).float()
        

        