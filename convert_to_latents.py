import argparse
import os
import glob
import torch
import numpy as np
import awkward as ak
from typing import Dict, List
from vae import NT_VAE
from dataloaders.prometheus import reduce_by_window
from dataloaders.data_utils import variable_length_collate_fn

def compute_summary_stats(grouped_times_np: np.ndarray, grouped_charges_np: np.ndarray) -> np.ndarray:
    """
    Computes 9 summary statistics from grouped pulse times and charges.
    Assumes grouped_times_np is sorted.
    """
    num_pulses = len(grouped_times_np)
    # Initialize with NaNs for time-based, 0 for charge-based if no pulses
    stats = np.full(9, np.nan, dtype=np.float32)

    if num_pulses == 0:
        stats[0] = 0.0  # total_charge
        stats[1] = 0.0  # charge_within_100ns
        stats[2] = 0.0  # charge_within_500ns
        # Other time-based stats (3-8) remain NaN
        return stats

    # 1. Total charge
    total_charge = np.sum(grouped_charges_np)
    stats[0] = total_charge
    
    # 4. Time of first pulse
    time_first_pulse = grouped_times_np[0]
    stats[3] = time_first_pulse

    # 7. Time of last pulse
    stats[6] = grouped_times_np[-1]

    # Handle zero total charge for subsequent calculations that depend on it
    if total_charge <= 1e-9: # Use a small epsilon for float comparison
        stats[1] = 0.0  # charge_within_100ns (already 0 if no pulses, but good to be explicit)
        stats[2] = 0.0  # charge_within_500ns
        # Time quantiles, mean, std will remain NaN as they are ill-defined
        return stats
        
    # 2. Charge within 100 ns of the first pulse
    mask_100ns = grouped_times_np <= (time_first_pulse + 100.0)
    stats[1] = np.sum(grouped_charges_np[mask_100ns])

    # 3. Charge within 500 ns of the first pulse
    mask_500ns = grouped_times_np <= (time_first_pulse + 500.0)
    stats[2] = np.sum(grouped_charges_np[mask_500ns])

    # Cumulative charges for quantile calculations
    cumulative_charges = np.cumsum(grouped_charges_np)

    # 5. Time at which 20% of the charge is collected
    q20_threshold = 0.20 * total_charge
    # Find first index where cumulative charge >= threshold
    q20_indices = np.where(cumulative_charges >= q20_threshold - 1e-9)[0] # Add epsilon for float comparison
    if len(q20_indices) > 0:
        stats[4] = grouped_times_np[q20_indices[0]]
    elif num_pulses > 0 : # If threshold not met but pulses exist, use last pulse time
        stats[4] = grouped_times_np[-1]


    # 6. Time at which 50% of the charge is collected
    q50_threshold = 0.50 * total_charge
    q50_indices = np.where(cumulative_charges >= q50_threshold - 1e-9)[0] # Add epsilon
    if len(q50_indices) > 0:
        stats[5] = grouped_times_np[q50_indices[0]]
    elif num_pulses > 0 : # If threshold not met but pulses exist, use last pulse time
        stats[5] = grouped_times_np[-1]


    # 8. Charge weighted mean of pulse arrival times
    charge_weighted_mean_time = np.sum(grouped_times_np * grouped_charges_np) / total_charge
    stats[7] = charge_weighted_mean_time

    # 9. Charge weighted standard deviation of pulse arrival times
    # Ensure total_charge is not zero before division for variance calculation
    if total_charge > 1e-9: # Check against a small epsilon
        weighted_variance = np.sum(grouped_charges_np * (grouped_times_np - charge_weighted_mean_time)**2) / total_charge
        stats[8] = np.sqrt(np.maximum(0, weighted_variance)) # Ensure variance is non-negative
    # else stats[8] remains NaN
    
    return stats

def preprocess_sensor_hits(
    raw_t_for_sensor: np.ndarray,
    sensor_pos_for_vae: torch.Tensor,
    grouping_window_ns: float,
    max_seq_len_padding: int,
    epsilon: float
) -> Dict:
    """
    Preprocesses hit data for a single sensor.
    
    Args:
        raw_t_for_sensor: Numpy array of hit times for this sensor.
        sensor_pos_for_vae: Scaled sensor position tensor for VAE input.
        grouping_window_ns: Time window for grouping hits.
        max_seq_len_padding: Maximum sequence length after padding for VAE.
        epsilon: Epsilon value for log normalization of times for VAE.
        
    Returns:
        Dictionary with preprocessed tensors for VAE and summary stats.
    """
    if len(raw_t_for_sensor) == 0:
        return {
            "charges_log_norm": torch.empty(0, dtype=torch.float32),
            "times_log_norm": torch.empty(0, dtype=torch.float32),
            "sensor_pos": sensor_pos_for_vae, # Still pass the sensor pos
            "summary_stats": compute_summary_stats(np.array([]), np.array([])),
        }

    # Apply time-window grouping
    grp_t_np, grp_c_np = reduce_by_window(raw_t_for_sensor, grouping_window_ns)
    
    # Compute summary statistics from the *full* grouped data (before VAE truncation)
    summary_stats_vec = compute_summary_stats(grp_t_np, grp_c_np)

    # Convert to tensors for VAE input processing
    rt_for_vae = torch.from_numpy(grp_t_np.copy()) # Use copy for VAE processing
    rc_for_vae = torch.from_numpy(grp_c_np.copy()) # Use copy for VAE processing
    
    # Handle sequence length truncation with count-weighted sampling (for VAE input)
    if max_seq_len_padding is not None and rt_for_vae.numel() > max_seq_len_padding:
        # Count-weighted random sampling
        probs = rc_for_vae.float() / rc_for_vae.sum().clamp(min=1e-9)
        if rc_for_vae.sum() <= 1e-9:  # If all charges are zero or empty
            idx = torch.randperm(rt_for_vae.numel())[:max_seq_len_padding]
        else:
            idx = torch.multinomial(probs, max_seq_len_padding, replacement=False)
        
        # Sort indices to preserve time order
        idx, _ = torch.sort(idx)
        rt_for_vae = rt_for_vae[idx]
        rc_for_vae = rc_for_vae[idx]
    
    # Handle empty sequences after VAE-specific processing
    if rt_for_vae.numel() == 0:
        # Still return summary_stats, but VAE inputs are empty
        return {
            "charges_log_norm": torch.empty(0, dtype=torch.float32),
            "times_log_norm": torch.empty(0, dtype=torch.float32),
            "sensor_pos": sensor_pos_for_vae, # sensor_pos is still valid
            "summary_stats": summary_stats_vec,
        }
    
    # Log-normalized charges (for VAE)
    charges_log_norm_vae = torch.log1p(rc_for_vae.clamp(min=0.0))
    
    # Log-normalized ABSOLUTE times (for VAE)
    times_log_norm_vae = torch.log(rt_for_vae.clamp(min=0.0) + epsilon)
    
    return {
        "charges_log_norm": charges_log_norm_vae,
        "times_log_norm": times_log_norm_vae,
        "sensor_pos": sensor_pos_for_vae,
        "summary_stats": summary_stats_vec, # Include the computed summary stats
    }

# process_batch function is removed as encoding is now per-sensor.
# extract_original_sensor_pos is removed as sensor positions are handled in main loop.

def main():
    parser = argparse.ArgumentParser(description='Convert events to latent space using the current om2vec model.')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Path to directory containing input parquet files.')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to directory for output parquet files.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint file.')
    parser.add_argument('--grouping_window_ns', type=float, default=2.0,
                       help='Time window in ns for grouping hits (default: 2.0).')
    parser.add_argument('--max_seq_len_padding', type=int, default=1024,
                       help='Maximum sequence length after padding (default: 1024). Set to 0 or negative for no limit.')
    parser.add_argument('--log_epsilon', type=float, default=1e-9,
                       help='Epsilon value for log normalization of times (default: 1e-9).')
    parser.add_argument('--chunk_size', type=int, default=1000, # Adjusted default chunk size for potentially more records
                       help='Number of events per output chunk.')
    
    args = parser.parse_args()
    
    # Validate max_seq_len_padding
    if args.max_seq_len_padding <= 0:
        args.max_seq_len_padding = None
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    model = NT_VAE.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input files
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not input_files:
        print(f"No parquet files found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} parquet files to process")
    
    # Initialize result containers
    chunk_iter = 0
    output_event_records = [] # Changed from dict of lists to list of dicts
    
    # Process each file
    for file_idx, file_path in enumerate(input_files):
        print(f"Processing file {file_idx + 1}/{len(input_files)}: {file_path}")
        
        try:
            # Load file data
            data_ak = ak.from_parquet(file_path)
            print(f"  Loaded {len(data_ak)} events from file, converting to Python list...")
            # Convert all events to a list of Python dictionaries immediately after loading
            events_py_list = ak.to_list(data_ak)
            data_ak = None # Free memory from awkward array if possible
            print(f"  Conversion complete. Processing {len(events_py_list)} events.")
            
            for event_idx, original_event_py_dict in enumerate(events_py_list):
                if (event_idx + 1) % 100 == 0:
                    print(f"    Processing event {event_idx + 1}/{len(events_py_list)}")

                # Access fields using dictionary .get() method for safety
                original_mc_truth = original_event_py_dict.get('mc_truth')
                # original_photons_field will now be a Python list of dicts, or None
                original_photons_field_py_list = original_event_py_dict.get('photons')
                
                om2vec_records_for_event = []

                if original_photons_field_py_list is not None and len(original_photons_field_py_list) > 0:
                    sensor_hits_map = {}
                    for photon_hit_py_dict in original_photons_field_py_list:
                        # Ensure all components of sensor_pos are present in the dictionary
                        if not all(k in photon_hit_py_dict for k in ['sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 't']):
                            # print(f"Skipping photon hit dict due to missing keys: {photon_hit_py_dict}")
                            continue
                        try:
                            sx = float(photon_hit_py_dict['sensor_pos_x'])
                            sy = float(photon_hit_py_dict['sensor_pos_y'])
                            sz = float(photon_hit_py_dict['sensor_pos_z'])
                            t_val = float(photon_hit_py_dict['t'])
                            sensor_coord_tuple = (sx, sy, sz)
                            sensor_hits_map.setdefault(sensor_coord_tuple, []).append(t_val)
                        except (TypeError, ValueError) as e:
                            # print(f"Warning: Could not convert photon hit data from dict {photon_hit_py_dict}: {e}")
                            continue # Skip this problematic hit

                    # Process data for each unique sensor in the event
                    for sensor_coord_tuple, hit_times_list in sensor_hits_map.items():
                        original_sx, original_sy, original_sz = sensor_coord_tuple
                        raw_t_for_sensor = np.array(sorted(hit_times_list), dtype=np.float32) # Ensure sorted times

                        sensor_pos_for_vae = torch.tensor(
                            [original_sx, original_sy, original_sz], dtype=torch.float32
                        ) / 1000.0  # Convert to km

                        preprocessed_sensor_dict = preprocess_sensor_hits(
                            raw_t_for_sensor,
                            sensor_pos_for_vae,
                            args.grouping_window_ns,
                            args.max_seq_len_padding,
                            args.log_epsilon
                        )

                        if preprocessed_sensor_dict["charges_log_norm"].numel() == 0:
                            # No valid VAE data for this sensor, but we might still want to record its position and empty stats
                            # For now, let's create a record with NaN latents if desired, or skip
                            # To match previous behavior of skipping, we continue
                            # If you want to include it with NaN latents, create the record here.
                            # For now, we skip if no VAE input can be formed.
                            # However, summary_stats are computed before VAE truncation, so they might be valid.
                            # Let's include sensors if they have summary stats, even if VAE input is empty.
                            if np.all(np.isnan(preprocessed_sensor_dict["summary_stats"])): # if all summary stats are NaN (e.g. no hits at all for sensor)
                                continue # Truly nothing to record for this sensor

                        # Prepare single sensor data for model (batch of 1)
                        # Ensure all required keys are present for collate_fn, even if empty for VAE
                        keys_for_collate = ["charges_log_norm", "times_log_norm", "sensor_pos", "summary_stats"]
                        collate_input_dict = {k: preprocessed_sensor_dict[k] for k in keys_for_collate if k in preprocessed_sensor_dict}
                        
                        # Add dummy attention_mask if not present (collate_fn might expect it)
                        # The collate_fn creates it based on charges_log_norm_padded, so it should be fine.

                        latent_vector = np.full(model.latent_dim, np.nan, dtype=np.float32) # Default to NaN

                        if preprocessed_sensor_dict["charges_log_norm"].numel() > 0 : # Only run VAE if there's data
                            single_sensor_collated_input = variable_length_collate_fn([collate_input_dict])
                            
                            for key_to_move in single_sensor_collated_input:
                                if isinstance(single_sensor_collated_input[key_to_move], torch.Tensor):
                                    single_sensor_collated_input[key_to_move] = single_sensor_collated_input[key_to_move].to(device)
                            
                            with torch.no_grad():
                                mu_full, _, _, _ = model.encode(
                                    single_sensor_collated_input["charges_log_norm_padded"],
                                    single_sensor_collated_input["times_log_norm_padded"],
                                    single_sensor_collated_input["sensor_pos_batched"],
                                    single_sensor_collated_input["attention_mask"]
                                )
                                latent_vector = mu_full.cpu().numpy().squeeze(axis=0) # Squeeze batch dim

                        om2vec_sensor_record = {
                            'sensor_pos_x': float(original_sx),
                            'sensor_pos_y': float(original_sy),
                            'sensor_pos_z': float(original_sz),
                            'latents': latent_vector,
                            'summary_stats': preprocessed_sensor_dict["summary_stats"]
                        }
                        om2vec_records_for_event.append(om2vec_sensor_record)
                
                # Construct the final output structure for this event
                output_event_structure_for_file = {
                    'mc_truth': original_mc_truth,
                    'photons': original_photons_field_py_list, # Store the Python list of dicts
                    'om2vec': om2vec_records_for_event # List of per-sensor om2vec data
                }
                output_event_records.append(output_event_structure_for_file)

                # Save chunk if needed (based on number of *events* processed)
                if len(output_event_records) >= args.chunk_size:
                    output_file = os.path.join(args.output_dir, f"om2vec_chunk_{chunk_iter}.parquet")
                    # Ensure awkward array handles list of lists for om2vec field correctly
                    ak.to_parquet(ak.Array(output_event_records), output_file)
                    print(f"  Saved chunk {chunk_iter} with {len(output_event_records)} events to {output_file}")
                    chunk_iter += 1
                    output_event_records = []
                    
        except Exception as e:
            print(f"  Error processing file {file_path}: {e}")
            continue
    
    # Save final chunk
    if output_event_records:
        output_file = os.path.join(args.output_dir, f"om2vec_chunk_{chunk_iter}.parquet")
        ak.to_parquet(ak.Array(output_event_records), output_file)
        print(f"Saved final chunk {chunk_iter} with {len(output_event_records)} events to {output_file}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()