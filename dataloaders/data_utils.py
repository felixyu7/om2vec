import torch
import numpy as np
import glob
import os
import random
from torch import Tensor
from torch.utils.data import Sampler, Dataset
from typing import List, Tuple, Optional, Union

def get_file_names(data_dirs: List[str], ranges: List[List[int]], shuffle_files: bool = False) -> List[str]:
    """
    Get file names from directories within specified ranges.
    
    Args:
        data_dirs: List of directories to search for files
        ranges: List of [start, end] ranges for each directory
        
    Returns:
        List of file paths
    """
    filtered_files = []
    for i, directory in enumerate(data_dirs):
        all_files = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        if shuffle_files:
            random.shuffle(all_files)
        file_range = ranges[i]
        filtered_files.extend(
            all_files[file_range[0]:file_range[1]]
        )
        
    if shuffle_files:
        # now shuffle the combined list
        random.shuffle(filtered_files)

    return filtered_files

class ParquetFileSampler(Sampler):
    """
    Custom sampler for parquet files that respects file boundaries during batching.
    
    This sampler first selects files in random order, then for each file,
    it shuffles the indices and yields batches from that file before moving
    to the next file.
    """
    def __init__(self, data_source: Dataset, cumulative_lengths: np.ndarray, batch_size: int):
       super().__init__(data_source) # Call Sampler's __init__
       self.data_source = data_source
       self.cumulative_lengths = cumulative_lengths  # expects array starting with 0, then cumulative sums
       self.batch_size = batch_size

    def __iter__(self):
        n_files = len(self.cumulative_lengths) - 1
        file_order = np.random.permutation(n_files)
        
        for file_index in file_order:
            start_idx = self.cumulative_lengths[file_index]
            end_idx = self.cumulative_lengths[file_index + 1]
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            for i in range(0, len(indices), self.batch_size):
                yield from indices[i:i+self.batch_size].tolist()

    def __len__(self) -> int:
       return len(self.data_source)

class IrregularDataCollator(object):
    """
    Collator for om2vec project.
    
    Handles batching of event data, where each event contains padded OM data.
    The __getitem__ from the dataset returns a dictionary of tensors.
    This collator will stack these tensors along a new batch dimension.
    """
    def __init__(self):
        pass
        
    def __call__(self, batch: List[dict]) -> dict:
        """
        Collates a list of dictionaries (each representing an event) into a single dictionary of batched tensors.

        Args:
            batch (List[dict]): A list of dictionaries, where each dictionary is an output
                                 from IceCube_Parquet_Dataset.__getitem__.
                                 Expected keys: "all_om_hits", "all_om_sensor_pos",
                                                "om_mask", "hit_mask", "label", "event_id".

        Returns:
            dict: A dictionary where each key maps to a batched tensor.
                  E.g., "all_om_hits" will be shape (batch_size, max_oms, max_photons, 2).
        """
        if not batch:
            return {}

        # Get keys from the first item, assuming all items have the same structure
        keys = batch[0].keys()
        collated_batch = {key: [] for key in keys}

        for item in batch:
            for key in keys:
                collated_batch[key].append(item[key])
        
        # Stack tensors for each key
        for key in keys:
            # Ensure all elements for a key are tensors before stacking
            # This is important if some __getitem__ could return non-tensor (e.g. None for optional fields)
            # However, our current __getitem__ returns tensors for all defined keys.
            elements = [torch.as_tensor(elem) if not torch.is_tensor(elem) else elem for elem in collated_batch[key]]
            collated_batch[key] = torch.stack(elements, dim=0)
            
        return collated_batch
        
def batched_coordinates(list_of_coords: List[Tensor],
                      dtype: Optional[torch.dtype] = None, 
                      device: Optional[Union[torch.device, str]] = None, 
                      requires_grad: bool = False) -> Tensor:
    """
    Convert a list of coordinate tensors to a batched tensor with batch indices.
    
    Args:
        list_of_coords: List of tensors, each of shape [Mi, D]
        dtype: Output data type
        device: Output device
        requires_grad: Whether output requires gradients
        
    Returns:
        Batched coordinates tensor of shape [sum(Mi), D+1] with batch indices prepended
    """
    # Infer the backend (NumPy or PyTorch) from the first element
    first = list_of_coords[0]
    # is_torch = torch.is_tensor(first) # No longer needed, assume torch tensors

    # if is_torch: # Assume torch tensor input
    # If no dtype provided, use the dtype of the first tensor
    if dtype is None:
        dtype = first.dtype
    # If no device provided, use the device of the first tensor
    if device is None:
        device = first.device

    # Collect the list of (batch_index, coordinates) pairs
    cat_list = []
    for b, coords in enumerate(list_of_coords):
        # Convert coords to desired dtype/device if needed
        coords = coords.to(device=device, dtype=dtype)

        # Create a column of batch indices
        b_idx = torch.full((coords.shape[0], 1), b, dtype=dtype, device=device,
                           requires_grad=requires_grad)

        # Concatenate [batch_index_column, coords]
        out_coords = torch.cat((b_idx, coords), dim=1)
        cat_list.append(out_coords)

    # Final concatenation
    out = torch.cat(cat_list, dim=0)
    # If the user wants gradient tracking on the final tensor:
    if requires_grad and not out.requires_grad:
        out.requires_grad_(True)

    return out