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

def preprocess_event(event_data, grouping_window_ns: float, max_seq_len_padding: int, epsilon: float) -> Dict:
    """
    Preprocess a single event using the same logic as the training pipeline.
    
    Args:
        event_data: Single event from awkward array
        grouping_window_ns: Time window for grouping hits
        max_seq_len_padding: Maximum sequence length after padding
        epsilon: Epsilon value for log normalization of times
        
    Returns:
        Dictionary with preprocessed tensors ready for model input
    """
    # Extract parameters from config
    # Parameters are now passed directly
    
    # Check if event has photons/hits
    if hasattr(event_data, 'photons') and hasattr(event_data.photons, 't'):
        raw_t = np.asarray(event_data.photons.t, dtype=np.float32)
    elif hasattr(event_data, 'hits_t'):
        raw_t = np.asarray(event_data.hits_t, dtype=np.float32)
    else:
        # Return empty tensors for events without hits
        return {
            "charges_log_norm": torch.empty(0, dtype=torch.float32),
            "times_log_norm": torch.empty(0, dtype=torch.float32),
            "sensor_pos": torch.zeros(3, dtype=torch.float32),
            "summary_stats": compute_summary_stats(np.array([]), np.array([])), # Default for no hits
        }
    
    # Skip events with no hits
    if len(raw_t) == 0:
        return {
            "charges_log_norm": torch.empty(0, dtype=torch.float32),
            "times_log_norm": torch.empty(0, dtype=torch.float32),
            "sensor_pos": torch.zeros(3, dtype=torch.float32),
            "summary_stats": compute_summary_stats(np.array([]), np.array([])), # Default for no hits
        }
    
    # Extract sensor position - try different field names for compatibility
    sensor_pos_x = float(event_data.sensor_pos_x)
    sensor_pos_y = float(event_data.sensor_pos_y) 
    sensor_pos_z = float(event_data.sensor_pos_z)
    
    sensor_pos = torch.tensor([sensor_pos_x, sensor_pos_y, sensor_pos_z], dtype=torch.float32) / 1000.0  # Convert to km
    
    # Apply time-window grouping using the same function as training pipeline
    grp_t_np, grp_c_np = reduce_by_window(raw_t, grouping_window_ns)
    
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
            "sensor_pos": sensor_pos, # sensor_pos is still valid
            "summary_stats": summary_stats_vec,
        }
    
    # Log-normalized charges (for VAE)
    charges_log_norm_vae = torch.log1p(rc_for_vae.clamp(min=0.0))
    
    # Log-normalized ABSOLUTE times (for VAE)
    times_log_norm_vae = torch.log(rt_for_vae.clamp(min=0.0) + epsilon)
    
    return {
        "charges_log_norm": charges_log_norm_vae,
        "times_log_norm": times_log_norm_vae,
        "sensor_pos": sensor_pos,
        "summary_stats": summary_stats_vec, # Include the computed summary stats
    }

def extract_original_sensor_pos(event_data) -> tuple[float, float, float]:
    """Extracts original, unscaled sensor positions from event data."""
    sensor_pos_x = 0.0
    sensor_pos_y = 0.0
    sensor_pos_z = 0.0
    if hasattr(event_data, 'sensor_pos_x'):
        sensor_pos_x = float(event_data.sensor_pos_x)
    if hasattr(event_data, 'sensor_pos_y'):
        sensor_pos_y = float(event_data.sensor_pos_y)
    if hasattr(event_data, 'sensor_pos_z'):
        sensor_pos_z = float(event_data.sensor_pos_z)
    # If top-level sensor_pos_x,y,z are not present, they default to 0.0.
    # The request "om2vec.sensor_pos_ are from the originals" implies event-level.
    return sensor_pos_x, sensor_pos_y, sensor_pos_z

def process_batch(batch_data: List[Dict], model, device) -> np.ndarray:
    """
    Process a batch of preprocessed events through the model.
    
    Args:
        batch_data: List of preprocessed event dictionaries
        model: Loaded NT_VAE model
        device: Device for computation
        
    Returns:
        Numpy array of latent vectors
    """
    if not batch_data:
        return np.array([])
    
    # Use the same collate function as training pipeline
    collated_batch = variable_length_collate_fn(batch_data)
    
    # Move to device
    for key in collated_batch:
        if isinstance(collated_batch[key], torch.Tensor):
            collated_batch[key] = collated_batch[key].to(device)
    
    # Run through model encoder
    with torch.no_grad():
        mu_full, logvar_full, _, _ = model.encode(
            collated_batch["charges_log_norm_padded"],
            collated_batch["times_log_norm_padded"], 
            collated_batch["sensor_pos_batched"],
            collated_batch["attention_mask"]
        )
        
        # Use mu (mean) as the latent representation
        # For deterministic latents, could also use reparameterize for sampling
        latents = mu_full.cpu().numpy()
    
    return latents

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
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for processing events.')
    parser.add_argument('--chunk_size', type=int, default=5000,
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
            data = ak.from_parquet(file_path)
            print(f"  Loaded {len(data)} events from file")
            
            # Process events in batches
            current_batch_preprocessed_for_model = []
            current_batch_original_events = []
            
            for event_idx, original_event_record in enumerate(data):
                # Preprocess event
                preprocessed_for_model = preprocess_event(original_event_record, args.grouping_window_ns, args.max_seq_len_padding, args.log_epsilon)
                
                # Skip empty events (those that result in no charges after preprocessing)
                if preprocessed_for_model["charges_log_norm"].numel() == 0:
                    continue
                
                current_batch_preprocessed_for_model.append(preprocessed_for_model)
                current_batch_original_events.append(original_event_record) # Store original awkward record
                
                # Process batch when full
                if len(current_batch_preprocessed_for_model) >= args.batch_size:
                    latents_for_batch = process_batch(current_batch_preprocessed_for_model, model, device)
                    
                    for i, latent_vec in enumerate(latents_for_batch):
                        original_event = current_batch_original_events[i]
                        preprocessed_data_for_event = current_batch_preprocessed_for_model[i]
                        summary_stats_vector = preprocessed_data_for_event["summary_stats"]

                        orig_sensor_x, orig_sensor_y, orig_sensor_z = extract_original_sensor_pos(original_event)
                        
                        output_record = {
                            'mc_truth': getattr(original_event, 'mc_truth', None),
                            'photons': getattr(original_event, 'photons', None),
                            'om2vec': {
                                'sensor_pos_x': orig_sensor_x,
                                'sensor_pos_y': orig_sensor_y,
                                'sensor_pos_z': orig_sensor_z,
                                'latents': latent_vec,
                                'summary_stats': summary_stats_vector
                            }
                        }
                        output_event_records.append(output_record)
                    
                    current_batch_preprocessed_for_model = []
                    current_batch_original_events = []
                    
                    # Save chunk if needed
                    if len(output_event_records) >= args.chunk_size:
                        output_file = os.path.join(args.output_dir, f"om2vec_chunk_{chunk_iter}.parquet")
                        ak.to_parquet(ak.Array(output_event_records), output_file)
                        print(f"  Saved chunk {chunk_iter} with {len(output_event_records)} events to {output_file}")
                        chunk_iter += 1
                        output_event_records = []
            
            # Process remaining batch
            if current_batch_preprocessed_for_model:
                latents_for_batch = process_batch(current_batch_preprocessed_for_model, model, device)
                
                for i, latent_vec in enumerate(latents_for_batch):
                    original_event = current_batch_original_events[i]
                    preprocessed_data_for_event = current_batch_preprocessed_for_model[i]
                    summary_stats_vector = preprocessed_data_for_event["summary_stats"]
                    
                    orig_sensor_x, orig_sensor_y, orig_sensor_z = extract_original_sensor_pos(original_event)

                    output_record = {
                        'mc_truth': getattr(original_event, 'mc_truth', None),
                        'photons': getattr(original_event, 'photons', None),
                        'om2vec': {
                            'sensor_pos_x': orig_sensor_x,
                            'sensor_pos_y': orig_sensor_y,
                            'sensor_pos_z': orig_sensor_z,
                            'latents': latent_vec,
                            'summary_stats': summary_stats_vector
                        }
                    }
                    output_event_records.append(output_record)
                    
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