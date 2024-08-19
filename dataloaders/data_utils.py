import glob
import os
import re

import torch
import numpy as np
from typing import Iterator, Optional, Sized

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