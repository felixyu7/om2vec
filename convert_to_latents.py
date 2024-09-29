import argparse, os, glob
import awkward as ak

import torch
import numpy as np
from vae import NT_VAE

def main(input_path, output_path, checkpoint_path, chunk_size=5000, max_time=6400, num_bins=6400):
    model = NT_VAE.load_from_checkpoint(checkpoint_path)
    model = model.eval()
    
    # check if cuda is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    chunk_iter = 0
    res = {'mc_truth': [], 'photons': [], 'latents': []}
    for file in files:
        print(f"Processing {file}")
        data = ak.from_parquet(file)
        for event in data:
            if len(event.photons.t) == 0:
                continue
            string_id = event.photons.string_id.to_numpy()
            sensor_id = event.photons.sensor_id.to_numpy()
            ts = event.photons.t.to_numpy() - event.photons.t.to_numpy().min()
            
            pos_t = np.array([
                string_id,
                sensor_id,
                ts
            ]).T
            
            # 8 unique OMs cut, adjustable as needed
            if np.unique(pos_t[:, :2], axis=0).shape[0] < 8:
                continue
            
            # Extract coordinates and times
            coordinates = pos_t[:, :2]
            times = pos_t[:, 2]

            # Get unique coordinates and indices
            unique_coords, indices = np.unique(coordinates, axis=0, return_inverse=True)

            # Create a list to hold corresponding times for each unique coordinate
            time_lists = [[] for _ in range(len(unique_coords))]

            # Populate the time lists based on unique coordinate indices
            for idx, time in zip(indices, times):
                time_lists[idx].append(time)

            # Convert the unique coordinates to a list of tuples
            unique_coords_list = [tuple(coord) for coord in unique_coords]
            
            event_binned_time_counts = []
            for dom_times in time_lists:
                # bin hits on individual sensors
                bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
                
                dom_times = np.array(dom_times)
                # do not consider hits with time > max_time
                max_time_mask = (dom_times < max_time)
                
                binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
                binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
                
                # put hits with time > max_time in the last bin
                binned_time_counts[-1] += np.sum(~max_time_mask)
                event_binned_time_counts.append(binned_time_counts)
                
            event_binned_time_counts = np.array(event_binned_time_counts)
            event_binned_time_counts = torch.from_numpy(event_binned_time_counts).float().to(device)
            inputs = torch.log(event_binned_time_counts + 1)
            latents, _ = model.encode(inputs)
            
            unique_coords_list = np.array(unique_coords_list)
            unique_coords_list = torch.from_numpy(unique_coords_list).float().to(device)
            latents = torch.hstack((unique_coords_list, latents)).cpu().detach().numpy()
            res['mc_truth'].append(event.mc_truth)
            res['photons'].append(event.photons)
            res['latents'].append(latents)
            if len(res['mc_truth']) >= chunk_size:
                ak.to_parquet(ak.Array(res), os.path.join(output_path, f"chunk_{chunk_iter}.parquet"))
                chunk_iter += 1
                res = {'mc_truth': [], 'photons': [], 'latents': []}
    ak.to_parquet(ak.Array(res), os.path.join(output_path, f"chunk_{chunk_iter}.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process events stored in Parquet files into latent space.')
    parser.add_argument('--input_dir', type=str, help='Path to the input Parquet files.')
    parser.add_argument('--output_dir', type=str, help='Path to save the output Parquet files.')
    parser.add_argument('--ckpt', type=str, help='Model checkpoint path.')
    parser.add_argument('--chunk_size', type=int, default=5000, help='Number of events to process at once.')
    parser.add_argument('--max_time', type=int, default=6400, help='Maximum time value.')
    parser.add_argument('--num_bins', type=int, default=6400, help='Number of bins for time histogram')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.ckpt)
    
    