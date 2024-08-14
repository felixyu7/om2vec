import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import pandas as pd
import gc, os, glob, re

from typing import Iterator, Optional, Sized
import polars
import pickle
from collections import OrderedDict
from bisect import bisect_right

class ICTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if self.cfg['training']:
            train_files = get_file_names(self.cfg['data_options']['train_data_files'], self.cfg['data_options']['train_data_file_ranges'])
            self.train_dataset = FileCache(cache_size=64,
                                           reduce_size=-1,
                                           resource_path='/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/GNN/software/src/nunet/resources',
                                           paths=train_files)
            
        valid_files = get_file_names(self.cfg['data_options']['valid_data_files'], self.cfg['data_options']['valid_data_file_ranges'])
        self.valid_dataset = FileCache(cache_size=64,
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
         
def get_meta_filename(fname):
    """
    Given a data file name, we simply find its corresponding metafile.
    """
    file_prefix = fname.split('.parquet')[0]
    return file_prefix + '.meta.parquet'

class FileCache(torch.utils.data.Dataset):
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
            # nevents_file = base_path + '/Nevents.pickle'  # TODO: be more smartly about this -PW
            # with open(nevents_file, "rb") as f:
            #     _Nevents = pickle.load(f)
            
            # Nevents = Nevents | _Nevents  # combine the dictionaries
            
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
        
        chunk_dict_1 = np.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_vae/data/22042_0_999_time_series_lengths.npy', allow_pickle=True).item()
        chunk_dict_2 = np.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_vae/data/22043_0_999_time_series_lengths.npy', allow_pickle=True).item()
        
        self.chunks = [chunk_dict_1[f] for f in paths if f in chunk_dict_1]
        self.chunks += [chunk_dict_2[f] for f in paths if f in chunk_dict_2]
        
        self.chunk_cumsum = np.cumsum(self.chunks)
        self.cache, self.meta_cache = None, None
        sensors = prepare_sensors(self.resource_path)
        self.geometry = torch.from_numpy(
            sensors[["x", "y", "z"]].values.astype(np.float32)
        )
        
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
            meta_fname = get_meta_filename(fname)
            
            df = polars.read_parquet(os.path.join(self.path_dict[fname], fname))
            # df_meta = polars.read_parquet(os.path.join(self.path_dict[fname], meta_fname))
            
            # group by each sensor
            df = df.group_by("event_id", "sensor_id").agg(
                [
                    polars.len(),
                    polars.col("time"),
                    polars.col("charge"),
                    polars.col("auxiliary"),
                ]
            )

            self.cache[fname] = df.sort("event_id")
            # self.meta_cache[fname] = df_meta
            if len(self.cache) > self.cache_size:
                del self.cache[list(self.cache.keys())[0]]
                # del self.meta_cache[list(self.meta_cache.keys())[0]]

    def __getitem__(self, idx0):
        fidx = bisect_right(self.chunk_cumsum, idx0)
        fname = self.files[fidx]
        idx = int(idx0 - self.chunk_cumsum[fidx - 1]) if fidx > 0 else idx0        
        
        self.load_data(fname)
        df = self.cache[fname][idx]

        # get minimum time of event for normalization
        min_time = np.concatenate(self.cache[fname].filter(polars.col("event_id") == df["event_id"][0])['time'].to_list()).min()

        sensor_id = df["sensor_id"][0]
        pos = self.geometry[sensor_id]
        
        # bin hits on individual sensors
        num_bins = 4096
        max_time = 4096 * 3 # 3ns time bins
        bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
        
        dom_times = df["time"][0].to_numpy() - min_time # shift by first hit time
        dom_charges = df["charge"][0].to_numpy()
        
        # do not consider hits with time > max_time
        max_time_mask = (dom_times < max_time)
        
        binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
        # binned_time_counts = np.histogram(binned_times, bins=bin_edges, weights=dom_charges[max_time_mask])[0]
        binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
        
        # put hits with time > max_time in the last bin
        # binned_time_counts[-1] += np.sum(dom_charges[~max_time_mask])
        binned_time_counts[-1] += np.sum(~max_time_mask)
        
        # normalize
        # binned_time_counts = np.log(binned_time_counts + 1)
        
        return torch.from_numpy(binned_time_counts).float(), pos.float()

class RandomChunkSampler(torch.utils.data.Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        chunks,
        num_samples: Optional[int] = None,
        generator=None,
        **kwargs,
    ) -> None:
        # chunks - a list of chunk sizes
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.chunks = chunks

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        cumsum = np.cumsum(self.chunks)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        chunk_list = torch.randperm(len(self.chunks), generator=generator).tolist()
        # sample indexes chunk by chunk
        yield_samples = 0
        for i in chunk_list:
            chunk_len = self.chunks[i]
            offset = cumsum[i - 1] if i > 0 else 0
            samples = (offset + torch.randperm(chunk_len, generator=generator)).tolist()
            if len(samples) <= self.num_samples - yield_samples:
                yield_samples += len(samples)
            else:
                samples = samples[: self.num_samples - yield_samples]
                yield_samples = self.num_samples
            yield from samples

    def __len__(self) -> int:
        return self.num_samples
    
def prepare_sensors(sensorpath):
    sensors = pd.read_csv(os.path.join(sensorpath, "sensor_geometry.csv")).astype(
        {
            "sensor_id": np.int16,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        }
    )
    sensors["string"] = 0
    sensors["qe"] = 0  # 1

    for i in range(len(sensors) // 60):
        start, end = i * 60, (i * 60) + 60
        sensors.loc[start:end, "string"] = i

        # High Quantum Efficiency in the lower 50 DOMs - https://arxiv.org/pdf/2209.03042.pdf (Figure 1)
        if i in range(78, 86):
            start_veto, end_veto = i * 60, (i * 60) + 10
            start_core, end_core = end_veto + 1, (i * 60) + 60
            sensors.loc[start_core:end_core, "qe"] = 1  # 1.35

    # https://github.com/graphnet-team/graphnet/blob/b2bad25528652587ab0cdb7cf2335ee254cfa2db/src/graphnet/models/detector/icecube.py#L33-L41
    # Assume that "rde" (relative dom efficiency) is equivalent to QE
    sensors["x"] /= 500
    sensors["y"] /= 500
    sensors["z"] /= 500

    return sensors

def get_file_names(data_dirs, ranges):
    filtered_files = []
    for i, directory in enumerate(data_dirs):
        all_files = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        file_range = ranges[i]
        filtered_files.extend(
            file for file in all_files
            if file_range[0] <= int(re.findall(r'\d+', os.path.basename(file))[-1]) <= file_range[1]
        )
    return sorted(filtered_files)