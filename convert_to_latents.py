import argparse, os, glob
import awkward as ak
import torch
import numpy as np
import yaml

# Import the model factory
from models import get_model

def main(config_path: str, input_path: str, output_path: str, checkpoint_path: str, 
         chunk_size: int = 5000, max_time: int = 6400, num_bins: int = 6400):
    """Main function to convert Prometheus data to latents using a specified model.
    
    Args:
        config_path: Path to the model configuration file (.yaml/.cfg).
        input_path: Directory containing input Parquet files.
        output_path: Directory to save output Parquet files with latents.
        checkpoint_path: Path to the model checkpoint (.ckpt).
        chunk_size: Number of events per output Parquet file.
        max_time: Maximum time (ns) for binning hits.
        num_bins: Number of bins for time histogram.
    """
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    with open(config_path, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # Override checkpoint path in config with the one provided via CLI
    cfg['checkpoint'] = checkpoint_path 

    # --- Load Model using Factory --- 
    # dataset_size is not needed for inference, pass None or -1
    try:
        # Pass config_path for potential architecture inference if needed
        model = get_model(cfg, dataset_size=None, cfg_path=config_path)
        model = model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Device Setup --- 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
    model = model.to(device)

    # --- Data Processing --- 
    files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    if not files:
        print(f"Error: No Parquet files found in {input_path}")
        return
        
    os.makedirs(output_path, exist_ok=True) # Ensure output directory exists
    chunk_iter = 0
    res = {'mc_truth': [], 'photons': [], 'latents': []}

    # Determine if model uses sensor positional encoding from loaded hparams
    # Use getattr to safely access hparams, provide default False
    use_pos_encoding = getattr(model.hparams, 'sensor_positional_encoding', False)
    print(f"Sensor positional encoding enabled: {use_pos_encoding}")

    # Pre-calculate bin edges
    bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)

    for file_idx, file in enumerate(files):
        print(f"Processing file {file_idx+1}/{len(files)}: {file}")
        try:
            data = ak.from_parquet(file)
        except Exception as e:
             print(f"  Error reading file {file}: {e}. Skipping.")
             continue

        for event_idx, event in enumerate(data):
            if len(event.photons.t) == 0:
                # print(f"  Event {event_idx}: Skipping empty event.")
                continue
            
            # Extract photon data
            string_ids = event.photons.string_id.to_numpy()
            sensor_ids = event.photons.sensor_id.to_numpy()
            times = event.photons.t.to_numpy() 
            
            # Check for NaN/inf times which can cause issues
            if not np.all(np.isfinite(times)):
                 print(f"  Event {event_idx}: Skipping event with non-finite times.")
                 continue
            
            min_time = np.min(times)
            relative_times = times - min_time # Shift times to start near 0
            
            # Combine coordinates and times
            coords_times = np.array([string_ids, sensor_ids, relative_times]).T
            
            # Filter unique sensors (OMs)
            unique_coords, unique_indices = np.unique(coords_times[:, :2], axis=0, return_inverse=True)
            
            # Optional: Minimum OMs cut (from original script)
            if len(unique_coords) < 8:
                # print(f"  Event {event_idx}: Skipping event with < 8 unique OMs ({len(unique_coords)} found).")
                continue

            # Prepare sensor positions if needed
            sensor_positions = None
            if use_pos_encoding:
                # Create positions based on unique_coords (string_id, sensor_id)
                # Need a way to map string/sensor ID to 3D coordinates if required by the model
                # Placeholder: Using string/sensor ID directly if model expects that.
                # Modify this if actual 3D coordinates are needed and available. 
                sensor_positions = torch.from_numpy(unique_coords).float().to(device) 
                # Add a dummy z-coordinate if model expects 3 features (x,y,z) 
                # Use getattr for safe access to hparams, which is a Namespace
                # Note: 'sensor_positional_encoding_dim' is not currently saved as a hyperparameter.
                # Defaulting to 3 if not found.
                pos_dim = getattr(model.hparams, 'sensor_positional_encoding_dim', 3)
                if pos_dim == 3 and sensor_positions.shape[1] == 2:
                     sensor_positions = torch.cat([sensor_positions, torch.zeros_like(sensor_positions[:, :1])], dim=1)

            # Bin times per unique sensor
            event_binned_counts = []
            for i in range(len(unique_coords)):
                sensor_times = relative_times[unique_indices == i]
                
                # Filter times exceeding max_time
                valid_time_mask = sensor_times < max_time
                times_to_bin = sensor_times[valid_time_mask]
                overflow_hits = np.sum(~valid_time_mask)
                
                # Bin valid times
                # Use histogram for potentially better performance/correctness
                binned_counts, _ = np.histogram(times_to_bin, bins=bin_edges)
                
                # Add overflow hits to the last bin
                if overflow_hits > 0:
                    binned_counts[-1] += overflow_hits
                
                event_binned_counts.append(binned_counts)
            
            # Convert to tensor and preprocess (log(1+x))
            event_binned_counts = np.array(event_binned_counts, dtype=np.float32)
            event_binned_counts_tensor = torch.from_numpy(event_binned_counts).to(device)
            inputs = torch.log(event_binned_counts_tensor + 1)

            # --- Get Latents from Model --- 
            with torch.no_grad(): # Ensure no gradients are computed
                # Use the unified encode interface
                # Pass sensor_positions only if needed
                encode_args = (inputs, sensor_positions) if use_pos_encoding else (inputs,)
                encoded_output = model.encode(*encode_args)

            # Handle potential tuple output (mu, logvar) from VAEs
            if isinstance(encoded_output, tuple):
                latents_tensor = encoded_output[0] # Use mu for VAEs
            else:
                latents_tensor = encoded_output # Use z for AEs
            
            # Combine unique coordinates with latents
            unique_coords_tensor = torch.from_numpy(unique_coords).float().to(device)
            combined_latents = torch.hstack((unique_coords_tensor, latents_tensor)).cpu().numpy()
            
            # Store results
            res['mc_truth'].append(event.mc_truth)
            res['photons'].append(event.photons)
            res['latents'].append(combined_latents)
            
            # Write chunk if size limit reached
            if len(res['mc_truth']) >= chunk_size:
                output_filename = os.path.join(output_path, f"chunk_{chunk_iter}.parquet")
                print(f"  Writing chunk {chunk_iter} to {output_filename}")
                try:
                    ak.to_parquet(ak.Array(res), output_filename)
                except Exception as e:
                    print(f"    Error writing chunk {chunk_iter}: {e}")
                chunk_iter += 1
                res = {'mc_truth': [], 'photons': [], 'latents': []} # Reset for next chunk

    # Write the final partial chunk if any data remains
    if len(res['mc_truth']) > 0:
        output_filename = os.path.join(output_path, f"chunk_{chunk_iter}.parquet")
        print(f"Writing final chunk {chunk_iter} to {output_filename}")
        try:
            ak.to_parquet(ak.Array(res), output_filename)
        except Exception as e:
            print(f"    Error writing final chunk {chunk_iter}: {e}")
            
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Prometheus events to latent representations using a specified model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration file (.yaml/.cfg).')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing input Parquet files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output Parquet files.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.ckpt).')
    parser.add_argument('--chunk_size', type=int, default=5000, help='Number of events per output Parquet file.')
    parser.add_argument('--max_time', type=int, default=6400, help='Maximum time (ns) for binning.')
    parser.add_argument('--num_bins', type=int, default=6400, help='Number of bins for time histogram.')

    args = parser.parse_args()
    
    main(config_path=args.config,
         input_path=args.input_dir, 
         output_path=args.output_dir, 
         checkpoint_path=args.ckpt, 
         chunk_size=args.chunk_size, 
         max_time=args.max_time, 
         num_bins=args.num_bins)
    
    