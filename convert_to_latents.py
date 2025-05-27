import argparse
import os
import glob
import awkward as ak
import numpy as np
import torch
import yaml

from vae import NT_VAE
from dataloaders.data_utils import reduce_by_window, variable_length_collate_fn

def main(input_path, output_path, checkpoint_path, grouping_window_ns=2.0, max_seq_len_padding=1024, epsilon=1e-8):
    grouping_window_ns = float(grouping_window_ns)
    max_seq_len_padding = int(max_seq_len_padding)
    epsilon = float(epsilon)

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
        data = ak.from_parquet(file)
        for event in data:
            # Check for photons field and required keys
            if not hasattr(event, "photons") or len(event.photons) == 0:
                continue
            photons = event.photons
            # Extract sensor positions and times
            try:
                pos_x = np.asarray(photons.sensor_pos_x)
                pos_y = np.asarray(photons.sensor_pos_y)
                pos_z = np.asarray(photons.sensor_pos_z)
                t = np.asarray(photons.t)
            except Exception:
                continue
            if len(t) == 0:
                continue
            # Group by unique sensor positions
            sensor_pos = np.stack([pos_x, pos_y, pos_z], axis=1)
            # Find unique sensors and group indices
            unique_pos, inv_idx = np.unique(sensor_pos, axis=0, return_inverse=True)
            # Prepare per-sensor hit times
            sensor_hits = [[] for _ in range(len(unique_pos))]
            for idx, time in zip(inv_idx, t):
                sensor_hits[idx].append(time)
            # For each sensor, aggregate hits by time window
            per_sensor_dicts = []
            for i, times in enumerate(sensor_hits):
                if len(times) == 0:
                    continue
                times = np.array(times, dtype=np.float32)
                # Use reduce_by_window to get representative times and counts
                rep_times, counts = reduce_by_window(times, grouping_window_ns)
                if len(rep_times) == 0:
                    continue
                # Truncate/pad if needed
                if len(rep_times) > max_seq_len_padding:
                    # Count-weighted random sample
                    probs = counts / np.clip(counts.sum(), 1e-9, None)
                    idx = np.random.choice(len(rep_times), max_seq_len_padding, replace=False, p=probs)
                    idx = np.sort(idx)
                    rep_times = rep_times[idx]
                    counts = counts[idx]
                # Pad if needed
                L = len(rep_times)
                if L < max_seq_len_padding:
                    pad_len = max_seq_len_padding - L
                    rep_times = np.pad(rep_times, (0, pad_len), constant_values=0)
                    counts = np.pad(counts, (0, pad_len), constant_values=0)
                # Log-normalize
                charges_log_norm = np.log1p(counts)
                times_log_norm = np.log(rep_times + epsilon)
                # Attention mask: True for padding
                attention_mask = np.zeros(max_seq_len_padding, dtype=bool)
                if L < max_seq_len_padding:
                    attention_mask[L:] = True
                # Sensor position (convert to km)
                sensor_pos_km = unique_pos[i] / 1000.0
                per_sensor_dicts.append({
                    "charges_log_norm": torch.tensor(charges_log_norm, dtype=torch.float32, device=device),
                    "times_log_norm": torch.tensor(times_log_norm, dtype=torch.float32, device=device),
                    "sensor_pos": torch.tensor(sensor_pos_km, dtype=torch.float32, device=device),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.bool, device=device)
                })
            if not per_sensor_dicts:
                continue
            # Batch sensors for this event using variable_length_collate_fn
            # (But all are already padded to max_seq_len_padding)
            batch = {
                "charges_log_norm": torch.stack([d["charges_log_norm"] for d in per_sensor_dicts]),
                "times_log_norm": torch.stack([d["times_log_norm"] for d in per_sensor_dicts]),
                "sensor_pos": torch.stack([d["sensor_pos"] for d in per_sensor_dicts]),
                "attention_mask": torch.stack([d["attention_mask"] for d in per_sensor_dicts])
            }
            # variable_length_collate_fn expects a list of dicts, but we already have padded tensors
            # So we can just pass through or mimic its output
            charges_log_norm_padded = batch["charges_log_norm"]
            times_log_norm_padded = batch["times_log_norm"]
            attention_mask = batch["attention_mask"]
            sensor_pos_batched = batch["sensor_pos"]
            # Feed to model.encode()
            with torch.no_grad():
                latents, _ = model.encode({
                    "charges_log_norm_padded": charges_log_norm_padded,
                    "times_log_norm_padded": times_log_norm_padded,
                    "attention_mask": attention_mask,
                    "sensor_pos_batched": sensor_pos_batched
                })
            # Move to CPU and numpy
            latents_np = latents.cpu().numpy()
            sensor_pos_np = sensor_pos_batched.cpu().numpy()
            # Save per event
            res['mc_truth'].append(event.mc_truth if hasattr(event, "mc_truth") else None)
            res['photons'].append(event.photons)
            # Save as list of dicts: each with sensor_pos and latent
            event_latents = []
            for j in range(latents_np.shape[0]):
                event_latents.append({
                    "sensor_pos_x": float(sensor_pos_np[j][0]),
                    "sensor_pos_y": float(sensor_pos_np[j][1]),
                    "sensor_pos_z": float(sensor_pos_np[j][2]),
                    "latents": latents_np[j].tolist()
                })
            res['om2vec'].append(event_latents)
            if len(res['mc_truth']) >= chunk_size:
                ak.to_parquet(ak.Array(res), os.path.join(output_path, f"chunk_{chunk_iter}.parquet"))
                chunk_iter += 1
                res = {'mc_truth': [], 'photons': [], 'om2vec': []}
    # Write any remaining
    if len(res['mc_truth']) > 0:
        ak.to_parquet(ak.Array(res), os.path.join(output_path, f"chunk_{chunk_iter}.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Prometheus photon hits to om2vec latents.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input Parquet files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output Parquet files.')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path.')
    parser.add_argument('--grouping_window_ns', type=float, default=2.0, help='Grouping window in ns (default: 2.0)')
    parser.add_argument('--max_seq_len_padding', type=int, default=1024, help='Max sequence length for padding/truncation (default: 1024)')
    parser.add_argument('--log_epsilon', type=float, default=1e-8, help='Epsilon for log normalization (default: 1e-8)')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.ckpt, args.grouping_window_ns, args.max_seq_len_padding, args.log_epsilon)