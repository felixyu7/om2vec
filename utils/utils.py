import torch
import numpy as np
import math

# --- Summary Statistics Calculation ---
def calculate_summary_statistics(t_values: np.ndarray, q_values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Calculates 10 summary statistics from a single sensor's list of (t,q) pairs.
    Input:
        t_values: NumPy array of raw photon arrival times for one sensor.
        q_values: NumPy array of raw photon charges for one sensor (assumed q=1 if not true charges).
        eps: Small epsilon to avoid log(0) or division by zero.
    Output:
        A 1D NumPy array of 10 summary statistics.
    """
    if len(t_values) == 0: # Handle empty pulse series
        return np.zeros(10, dtype=np.float32)

    # Ensure t_values are sorted for time-based statistics
    sort_indices = np.argsort(t_values)
    t_sorted = t_values[sort_indices]
    q_sorted_by_t = q_values[sort_indices]

    # 1. N_true: The actual number of photon pulses
    n_true = float(len(t_sorted))

    # 2. c_total: total DOM charge
    c_total = np.sum(q_sorted_by_t)

    # 3. c_500ns: charge within 500 ns of the first pulse
    t_first_val = t_sorted[0]
    c_500ns = np.sum(q_sorted_by_t[t_sorted < t_first_val + 500.0])

    # 4. c_100ns: charge within 100 ns of the first pulse
    c_100ns = np.sum(q_sorted_by_t[t_sorted < t_first_val + 100.0])

    # 5. t_first: time of first pulse
    # t_first_val is already defined

    # 6. t_last: time of the last pulse
    t_last_val = t_sorted[-1]

    # Calculate cumulative charge
    cumulative_charge = np.cumsum(q_sorted_by_t)
    
    # 7. t_20%: time at which 20% of the total charge is collected
    idx_20 = np.searchsorted(cumulative_charge, 0.20 * c_total + eps, side='left')
    t_20_val = t_sorted[min(idx_20, len(t_sorted)-1)] if c_total > eps else t_first_val

    # 8. t_50%: time at which 50% of the total charge is collected
    idx_50 = np.searchsorted(cumulative_charge, 0.50 * c_total + eps, side='left')
    t_50_val = t_sorted[min(idx_50, len(t_sorted)-1)] if c_total > eps else t_first_val
    
    # 9. t_mean: charge-weighted mean of the pulse arrival times
    t_mean_val = np.sum(t_sorted * q_sorted_by_t) / (c_total + eps) if c_total > eps else t_first_val

    # 10. t_std: charge-weighted standard deviation of the pulse arrival times
    if c_total > eps and n_true > 1:
        t_std_val = np.sqrt(np.sum(q_sorted_by_t * (t_sorted - t_mean_val)**2) / (c_total * (n_true -1) / n_true + eps) )
    elif c_total > eps and n_true == 1:
        t_std_val = 0.0 # Std dev is 0 for a single point
    else: # No charge or no pulses
        t_std_val = 0.0


    return np.array([
        n_true, c_total, c_500ns, c_100ns, t_first_val,
        t_last_val, t_20_val, t_50_val, t_mean_val, t_std_val
    ], dtype=np.float32)


# --- Fixed Normalization Functions ---
# For t, q values (individual photon characteristics)
def normalize_tq(value: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Log-normalizes t or q values: log(value + 1.0). Add eps for stability if values can be very close to -1."""
    return torch.log(value + 1.0 + eps)

def denormalize_tq(norm_value: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """De-normalizes t or q values: exp(norm_value) - 1.0."""
    return torch.exp(norm_value) - 1.0 - eps

# For sensor position (x, y, z)
def normalize_sensor_pos(pos_tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes sensor positions by dividing by 100.0."""
    return pos_tensor / 100.0

def denormalize_sensor_pos(norm_pos_tensor: torch.Tensor) -> torch.Tensor:
    """De-normalizes sensor positions by multiplying by 100.0."""
    return norm_pos_tensor * 100.0

# For the 10-element z_summary vector
# Indices for z_summary elements (0-indexed):
# 0: N_true (NOW log transform)
# 1: c_total (log transform)
# 2: c_500ns (log transform)
# 3: c_100ns (log transform)
# 4: t_first (log transform, assuming t relative or can be large)
# 5: t_last (log transform)
# 6: t_20% (log transform)
# 7: t_50% (log transform)
# 8: t_mean (log transform)
# 9: t_std (log transform)
# Note: If times are relative and can be 0, log(0+1) = 0 is fine.
# If times can be negative (e.g. relative to some trigger not always first),
# a shift might be needed before log, or a different transform.
# Assuming times are non-negative here based on typical photon data.

Z_SUMMARY_LOG_TRANSFORM_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # N_true (index 0) is now included

def normalize_z_summary(z_summary_tensor: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Normalizes the 10-element z_summary tensor.
    Applies log(x+1+eps) to ALL specified elements (including N_true at index 0).
    """
    if not isinstance(z_summary_tensor, torch.Tensor):
        z_summary_tensor = torch.as_tensor(z_summary_tensor, dtype=torch.float32)
    
    z_summary_norm = z_summary_tensor.clone()
    for idx in Z_SUMMARY_LOG_TRANSFORM_INDICES:
        if idx < z_summary_norm.shape[-1]: # Check if index is within bounds
             z_summary_norm[..., idx] = torch.log(z_summary_tensor[..., idx] + 1.0 + eps)
    return z_summary_norm

def denormalize_z_summary(z_summary_norm_tensor: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    De-normalizes the 10-element z_summary_norm tensor.
    Applies exp(x)-1-eps to ALL specified elements (including N_true at index 0).
    """
    if not isinstance(z_summary_norm_tensor, torch.Tensor):
        z_summary_norm_tensor = torch.as_tensor(z_summary_norm_tensor, dtype=torch.float32)

    z_summary_raw = z_summary_norm_tensor.clone()
    for idx in Z_SUMMARY_LOG_TRANSFORM_INDICES:
        if idx < z_summary_raw.shape[-1]: # Check if index is within bounds
            z_summary_raw[..., idx] = torch.exp(z_summary_norm_tensor[..., idx]) - 1.0 - eps
    return z_summary_raw


# --- Positional Encoding ---
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Not a model parameter, but part of state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if batch_first=False for Transformer
               or [batch_size, seq_len, embedding_dim] if batch_first=True.
               This implementation assumes Transformer's default batch_first=False,
               so input x is expected as [seq_len, batch_size, d_model].
               If your Transformer uses batch_first=True, x will be [batch, seq, dim]
               and you'll need to adjust: x = x + self.pe[:x.size(1)].transpose(0,1)
        """
        # Assuming x is [seq_len, batch_size, d_model] for nn.TransformerEncoder default
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEncodingBatchFirst(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_model/2]
        
        pe = torch.zeros(1, max_len, d_model) # [1, max_len, d_model] for batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True)
        """
        # x is [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)