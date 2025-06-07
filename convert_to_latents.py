import argparse
import os
import glob
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import awkward as ak
import numpy as np
import torch

from vae import NT_VAE
from dataloaders.prometheus import reduce_by_window

import time

def compute_summary_statistics(rep_times: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Compute 9 summary statistics from processed photon hit data (post reduce_by_window).
    
    Args:
        rep_times: Representative times from reduce_by_window
        counts: Counts/charges for each time bin from reduce_by_window
    
    Returns:
        np.ndarray: [total_charge, charge_100ns, charge_500ns, first_pulse_time,
                    time_20_percent, time_50_percent, last_pulse_time, 
                    charge_weighted_mean, charge_weighted_std]
    """
    if len(rep_times) == 0:
        return np.zeros(9, dtype=np.float32)
    
    # Total charge is sum of all counts
    total_charge = float(np.sum(counts))
    
    # Sort by time for easier processing
    sort_idx = np.argsort(rep_times)
    sorted_times = rep_times[sort_idx]
    sorted_counts = counts[sort_idx]
    
    first_pulse_time = sorted_times[0]
    last_pulse_time = sorted_times[-1]
    
    # Charge within time windows of first pulse
    mask_100ns = sorted_times <= (first_pulse_time + 100.0)
    mask_500ns = sorted_times <= (first_pulse_time + 500.0)
    charge_100ns = float(np.sum(sorted_counts[mask_100ns]))
    charge_500ns = float(np.sum(sorted_counts[mask_500ns]))
    
    # Cumulative charge for percentile calculations
    cumulative_charge = np.cumsum(sorted_counts)
    
    # Time at which 20% and 50% of charge is collected
    charge_20_threshold = 0.2 * total_charge
    charge_50_threshold = 0.5 * total_charge
    
    idx_20 = np.searchsorted(cumulative_charge, charge_20_threshold)
    idx_50 = np.searchsorted(cumulative_charge, charge_50_threshold)
    
    # Ensure indices are within bounds
    idx_20 = min(idx_20, len(sorted_times) - 1)
    idx_50 = min(idx_50, len(sorted_times) - 1)
    
    time_20_percent = sorted_times[idx_20]
    time_50_percent = sorted_times[idx_50]
    
    # Charge weighted mean and std
    if total_charge > 0:
        charge_weighted_mean = np.sum(sorted_times * sorted_counts) / total_charge
        charge_weighted_variance = np.sum(sorted_counts * (sorted_times - charge_weighted_mean)**2) / total_charge
        charge_weighted_std = np.sqrt(charge_weighted_variance)
    else:
        charge_weighted_mean = 0.0
        charge_weighted_std = 0.0
    
    return np.array([
        total_charge, charge_100ns, charge_500ns, first_pulse_time,
        time_20_percent, time_50_percent, last_pulse_time,
        charge_weighted_mean, charge_weighted_std
    ], dtype=np.float32)

def process_sensor_data(sensor_times: np.ndarray, grouping_window_ns: float, 
                       max_seq_len_padding: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimized sensor data processing with minimal allocations."""
    sensor_times = np.ascontiguousarray(sensor_times, dtype=np.float32)

    # Assuming reduce_by_window is now defined/imported as discussed above
    rep_times_orig, counts_orig = reduce_by_window(sensor_times, grouping_window_ns)

    # Pre-allocate output arrays
    # Padded time values remain 0.0, charges also 0.0 for padded part.
    final_norm_charges = np.zeros(max_seq_len_padding, dtype=np.float32)
    final_norm_times = np.zeros(max_seq_len_padding, dtype=np.float32)
    # Initialize attn_mask to all True (all masked). Actual data will set parts to False.
    final_attn_mask = np.ones(max_seq_len_padding, dtype=bool)

    if len(rep_times_orig) == 0:
        # If empty after grouping, return zero-filled arrays and all-True mask
        return final_norm_charges, final_norm_times, final_attn_mask, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Continue with non-empty processing
    rep_times, counts = rep_times_orig, counts_orig
    current_seq_len = len(rep_times)

    if current_seq_len > max_seq_len_padding:
        counts_sum = counts.sum()
        if counts_sum > 1e-9: # Script 2 clamps sum at 1e-9 for prob calculation
            probs = counts / counts_sum # np.maximum(counts,0) not strictly needed if counts always >=0
            selected_indices = np.random.choice(
                current_seq_len, max_seq_len_padding,
                replace=False, p=probs
            )
        else: # counts_sum <= 1e-9 (or zero) -> uniform random sampling
            selected_indices = np.random.permutation(current_seq_len)[:max_seq_len_padding]

        selected_indices.sort()
        rep_times = rep_times[selected_indices]
        counts = counts[selected_indices]
        current_seq_len = max_seq_len_padding

    # Fill actual data into the final pre-allocated arrays
    # Add clamping to match Script 2's robustness, assuming counts/rep_times should be non-negative
    final_norm_charges[:current_seq_len] = np.log1p(np.maximum(counts, 0))
    final_norm_times[:current_seq_len] = np.log(np.maximum(rep_times, 0) + epsilon)

    # Set attention mask for actual data part to False (attend to these)
    final_attn_mask[:current_seq_len] = False

    return final_norm_charges, final_norm_times, final_attn_mask, rep_times, counts


def main(input_path, output_path, checkpoint_path, grouping_window_ns=2.0, max_seq_len_padding=1024, epsilon=1e-9, inference_batch_size=256):
    # Convert parameters once
    grouping_window_ns = float(grouping_window_ns)
    max_seq_len_padding = int(max_seq_len_padding)
    epsilon = float(epsilon)
    inference_batch_size = int(inference_batch_size)

    # Load model
    model = NT_VAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    chunk_size = 5000
    chunk_iter = 0
    res = {'mc_truth': [], 'photons': [], 'om2vec': []}
    
    for file in files:
        print(f"Processing {file}")
        stime = time.time()
        data = ak.from_parquet(file)
        for event in data:
            # Check for photons field early
            if not hasattr(event, "photons"):
                continue
            
            photons = event.photons
            
            # Extract and convert to numpy arrays in one go
            pos_arrays = [np.asarray(getattr(photons, f"sensor_pos_{axis}"), dtype=np.float32) 
                         for axis in ['x', 'y', 'z']]
            pos_x, pos_y, pos_z = pos_arrays
            t = np.asarray(photons.t, dtype=np.float32)
            
            # Early check for empty data
            if len(t) == 0:
                continue
            
            # Stack positions efficiently
            sensor_coords = np.column_stack(pos_arrays)
            
            # Find unique sensors and group indices
            unique_coords, inv_idx = np.unique(sensor_coords, axis=0, return_inverse=True)
            n_unique_sensors = len(unique_coords)
            
            if n_unique_sensors == 0:
                continue
            
            # Pre-allocate lists with estimated capacity
            event_charges = []
            event_times = []
            event_positions = []
            event_attn_masks = []
            event_summary_stats = []
            
            # Process each unique sensor
            for sensor_idx in range(n_unique_sensors):
                sensor_mask = (inv_idx == sensor_idx)
                sensor_times = t[sensor_mask]
                
                if len(sensor_times) == 0:
                    continue
                
                # Process sensor data efficiently
                norm_charges, norm_times, attn_mask, rep_times, counts = process_sensor_data(
                    sensor_times, grouping_window_ns, max_seq_len_padding, epsilon
                )
                
                # Compute summary statistics from processed data (post reduce_by_window)
                summary_stats = compute_summary_statistics(rep_times, counts)

                event_charges.append(norm_charges)
                event_times.append(norm_times)
                event_positions.append(unique_coords[sensor_idx] * 0.001)  # Scale to km
                event_attn_masks.append(attn_mask)
                event_summary_stats.append(summary_stats)
            
            if not event_charges:
                continue
            
            # Convert to tensors efficiently
            n_sensors = len(event_charges)
            non_blocking = device.type == 'cuda'
            charges_batch = torch.from_numpy(np.stack(event_charges)).to(device, non_blocking=non_blocking)
            times_batch = torch.from_numpy(np.stack(event_times)).to(device, non_blocking=non_blocking)
            positions_batch = torch.from_numpy(np.stack(event_positions)).to(device, non_blocking=non_blocking)
            attn_mask_batch = torch.from_numpy(np.stack(event_attn_masks)).to(device, non_blocking=non_blocking)
            
            # Model inference with optimized batching
            with torch.inference_mode():
                if n_sensors <= inference_batch_size:
                    latents, _, _, _ = model.encode(charges_batch, times_batch, positions_batch, attn_mask_batch)
                else:
                    # Process in smaller batches
                    latents_list = []
                    for i in range(0, n_sensors, inference_batch_size):
                        end_idx = min(i + inference_batch_size, n_sensors)
                        batch_latents, _, _, _ = model.encode(
                            charges_batch[i:end_idx],
                            times_batch[i:end_idx], 
                            positions_batch[i:end_idx],
                            attn_mask_batch[i:end_idx]
                        )
                        latents_list.append(batch_latents)
                    latents = torch.cat(latents_list, dim=0)
            
            # Convert to numpy efficiently
            latents_np = latents.cpu().numpy()
            positions_np = positions_batch.cpu().numpy()
            summary_stats_np = np.stack(event_summary_stats)
            
            # Build output efficiently
            event_latents = [
                {
                    "sensor_pos_x": float(positions_np[j, 0]),
                    "sensor_pos_y": float(positions_np[j, 1]), 
                    "sensor_pos_z": float(positions_np[j, 2]),
                    "latents": latents_np[j].tolist(),
                    "summary_stats": summary_stats_np[j].tolist()
                }
                for j in range(n_sensors)
            ]
            
            # Store results
            res['mc_truth'].append(getattr(event, "mc_truth", None))
            res['photons'].append(event.photons)
            res['om2vec'].append(event_latents)
            
            # Write chunk if needed
            if len(res['mc_truth']) >= chunk_size:
                ak.to_parquet(ak.Array(res), os.path.join(output_path, f"chunk_{chunk_iter}.parquet"))
                chunk_iter += 1
                res = {'mc_truth': [], 'photons': [], 'om2vec': []}      

        print(f"Processed {file} in {time.time() - stime:.2f} seconds")
        
    # Write remaining data
    if len(res['mc_truth']) > 0:
        ak.to_parquet(ak.Array(res), os.path.join(output_path, f"chunk_{chunk_iter}.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Prometheus photon hits to om2vec latents.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input Parquet files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output Parquet files.')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path.')
    parser.add_argument('--grouping_window_ns', type=float, default=2.0, help='Grouping window in ns (default: 2.0)')
    parser.add_argument('--max_seq_len_padding', type=int, default=1024, help='Max sequence length for padding/truncation (default: 1024)')
    parser.add_argument('--log_epsilon', type=float, default=1e-9, help='Epsilon for log normalization (default: 1e-9)')
    parser.add_argument('--inference_batch_size', type=int, default=256, help='Batch size for model inference (default: 256)')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.ckpt, args.grouping_window_ns, args.max_seq_len_padding, args.log_epsilon, args.inference_batch_size)