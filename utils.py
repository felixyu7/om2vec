import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (d_model/2)
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe is now (max_len, d_model)
        # We want it to be (1, max_len, d_model) to easily add to (B, S, E)
        # Or, if Transformer is not batch_first, then (max_len, 1, d_model)
        # Since our Transformer is batch_first=True, input x to forward will be (B, S, E)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (if batch_first=True)
        """
        # x is (B, S, E). self.pe is (1, max_len, E)
        # We need to add self.pe[:, :S, :] to x
        # self.pe is (1, max_len, d_model). We need (1, seq_len, d_model)
        # x is (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def calculate_summary_statistics(raw_times_np: np.ndarray, raw_counts_np: np.ndarray) -> np.ndarray:
    """
    Calculates 9 summary statistics for a given photon distribution.

    Args:
        raw_times_np (np.ndarray): Array of pulse arrival times.
        raw_counts_np (np.ndarray): Array of pulse charges (counts).

    Returns:
        np.ndarray: A 1D array of 9 summary statistics.
                    Returns zeros if input is empty.
    """
    if raw_times_np.size == 0 or raw_counts_np.size == 0:
        return np.zeros(9, dtype=np.float32)

    # Ensure sorted by time for cumulative calculations and first/last pulse
    sort_indices = np.argsort(raw_times_np)
    times = raw_times_np[sort_indices]
    counts = raw_counts_np[sort_indices]

    # 1. Total charge
    total_charge = np.sum(counts)
    if total_charge == 0: # Avoid division by zero if all counts are zero
        return np.zeros(9, dtype=np.float32)

    # 4. Time of first pulse
    time_first_pulse = times[0]

    # 7. Time of last pulse
    time_last_pulse = times[-1]

    # 2. Charge within 100 ns of the first pulse
    charge_within_100ns = np.sum(counts[times <= time_first_pulse + 100.0])

    # 3. Charge within 500 ns of the first pulse
    charge_within_500ns = np.sum(counts[times <= time_first_pulse + 500.0])

    # 8. Charge-weighted mean of pulse arrival times
    charge_weighted_mean_time = np.sum(times * counts) / total_charge

    # 9. Charge-weighted standard deviation of pulse arrival times
    charge_weighted_std_time = np.sqrt(np.sum(((times - charge_weighted_mean_time)**2) * counts) / total_charge)

    # Cumulative charge calculation for 20% and 50% times
    cumulative_charge = np.cumsum(counts)
    
    # 5. Time at which 20% of the charge is collected
    target_charge_20 = 0.20 * total_charge
    idx_20_percent = np.searchsorted(cumulative_charge, target_charge_20, side='left')
    if idx_20_percent < len(times):
        time_charge_20_percent = times[idx_20_percent]
    else: # Should not happen if total_charge > 0 and cumulative_charge reaches total_charge
        time_charge_20_percent = time_last_pulse 

    # 6. Time at which 50% of the charge is collected
    target_charge_50 = 0.50 * total_charge
    idx_50_percent = np.searchsorted(cumulative_charge, target_charge_50, side='left')
    if idx_50_percent < len(times):
        time_charge_50_percent = times[idx_50_percent]
    else:
        time_charge_50_percent = time_last_pulse

    return np.array([
        total_charge,
        charge_within_100ns,
        charge_within_500ns,
        time_first_pulse,
        time_charge_20_percent,
        time_charge_50_percent,
        time_last_pulse,
        charge_weighted_mean_time,
        charge_weighted_std_time
    ], dtype=np.float32)