import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def imq_kernel_multi(z, prior_z, scales=None, eps=1e-7):
    if scales is None:
        scales = [0.1, 0.2, 0.5, 1., 2., 5., 10.]

    d2_z_z = torch.cdist(z,        z,        p=2).pow(2)
    d2_p_p = torch.cdist(prior_z,  prior_z,  p=2).pow(2)
    d2_z_p = torch.cdist(z,        prior_z,  p=2).pow(2)

    k_zz = k_pp = k_zp = 0.0
    dim = z.size(1)

    for s in scales:
        c = 2.0 * dim * s         # <<< heavier-tail constant
        k_zz += c / (c + d2_z_z / s + eps)
        k_pp += c / (c + d2_p_p / s + eps)
        k_zp += c / (c + d2_z_p / s + eps)

    # average over the 7 radii → stable gradients
    k_zz /= len(scales); k_pp /= len(scales); k_zp /= len(scales)

    # unbiased: drop diagonals
    k_zz -= torch.diag_embed(torch.diagonal(k_zz))
    k_pp -= torch.diag_embed(torch.diagonal(k_pp))
    return k_zz, k_pp, k_zp

# --- Mixture of Gaussians NLL helper ---
def mixture_gaussian_nll(target, weights, means, logvars, eps=1e-8):
    """
    Args:
        target: (N,) ground truth values
        weights: (N, K) mixture logits (not softmaxed)
        means: (N, K)
        logvars: (N, K)
    Returns:
        nll: scalar, mean negative log-likelihood
    """
    # weights: logits, so softmax
    log_weights = F.log_softmax(weights, dim=-1)  # (N, K)
    var = torch.exp(logvars)
    # log-prob for each component
    log_prob = -0.5 * (np.log(2 * np.pi) + logvars + ((target.unsqueeze(-1) - means) ** 2) / (var + eps))
    # log-sum-exp over components
    log_mix = torch.logsumexp(log_weights + log_prob, dim=-1)
    nll = -log_mix.mean()
    return nll

def calculate_sequence_lengths(attention_mask):
    """
    Calculates the actual sequence lengths from a boolean attention mask.

    Args:
        attention_mask (torch.Tensor): Boolean tensor of shape (B, S),
                                       where True indicates a padding token.

    Returns:
        torch.Tensor: Tensor of shape (B,) containing actual sequence lengths.
    """
    # If True is padding, then non-padding is (attention_mask == False) or (~attention_mask)
    # Summing along the sequence dimension gives the number of non-padding tokens.
    return (~attention_mask).sum(dim=1).long()

def calculate_summary_stats(charges_log_norm_padded, times_log_norm_padded, attention_mask, epsilon=1e-9):
    """
    Calculates various summary statistics from padded, log-normalized charge and absolute time sequences.

    Args:
        charges_log_norm_padded (torch.Tensor): (B, S) log1p normalized charges.
        times_log_norm_padded (torch.Tensor): (B, S) log normalized absolute times.
        attention_mask (torch.Tensor): (B, S) boolean, True for padding.
        epsilon (float): Small value for log stability.

    Returns:
        dict: Contains 'log_seq_length', 'log_total_charge', 
              'log_first_hit_time', 'log_last_hit_time', 'original_lengths'.
    """
    B, S = charges_log_norm_padded.shape
    device = charges_log_norm_padded.device

    original_lengths = calculate_sequence_lengths(attention_mask)
    log_seq_length = torch.log(original_lengths.float().clamp(min=1.0)) # Clamp for log(0)

    # Un-normalize charges (inverse of log1p)
    charges_unnorm = torch.exp(charges_log_norm_padded) - 1.0
    charges_unnorm.masked_fill_(attention_mask, 0) # Zero out padded values

    # Un-normalize times (inverse of log)
    times_unnorm = torch.exp(times_log_norm_padded)
    # For times, padded values should not affect min/max, so we can use a large/small number
    # or rely on masking after selection.
    
    # Total charge
    total_charge_unnorm = charges_unnorm.sum(dim=1)
    log_total_charge = torch.log1p(total_charge_unnorm.clamp(min=0.0))

    # First hit time
    # Mask padded times with a large value before finding min
    masked_times_for_first = times_unnorm.clone()
    masked_times_for_first.masked_fill_(attention_mask, float('inf'))
    first_hit_time_unnorm = torch.min(masked_times_for_first, dim=1).values
    # Handle cases where all are inf (empty sequence after masking, though original_lengths >= 1)
    first_hit_time_unnorm[first_hit_time_unnorm == float('inf')] = 0.0 
    log_first_hit_time = torch.log(first_hit_time_unnorm + epsilon)

    # Last hit time
    # Mask padded times with a small value (or 0) before finding max
    masked_times_for_last = times_unnorm.clone()
    masked_times_for_last.masked_fill_(attention_mask, 0.0) # Padded values won't be max
    last_hit_time_unnorm = torch.zeros(B, device=device)

    for i in range(B):
        if original_lengths[i] > 0:
            last_hit_time_unnorm[i] = masked_times_for_last[i, :original_lengths[i]].max()
        # else, it remains 0, log(epsilon) will be the result, which is reasonable for empty/padded seq

    log_last_hit_time = torch.log(last_hit_time_unnorm + epsilon)
    
    return {
        'log_seq_length': log_seq_length,
        'log_total_charge': log_total_charge,
        'log_first_hit_time': log_first_hit_time,
        'log_last_hit_time': log_last_hit_time,
        'original_lengths': original_lengths
    }

def convert_absolute_times_to_log_intervals(times_log_norm_padded, original_lengths, attention_mask, epsilon=1e-9):
    """
    Converts log-normalized absolute times to log-normalized time intervals.
    The input times_padded for the encoder should be these log_intervals.

    Args:
        times_log_norm_padded (torch.Tensor): (B, S) log normalized absolute times.
        original_lengths (torch.Tensor): (B,) actual sequence lengths.
        attention_mask (torch.Tensor): (B, S) boolean, True for padding.
        epsilon (float): Small value for log stability.

    Returns:
        torch.Tensor: (B, S) log-normalized time intervals, padded with 0.
    """
    B, S = times_log_norm_padded.shape
    device = times_log_norm_padded.device

    times_unnorm = torch.exp(times_log_norm_padded)
    
    # Initialize intervals tensor (e.g., with zeros)
    time_intervals_unnorm_padded = torch.zeros_like(times_unnorm)

    for i in range(B):
        seq_len = original_lengths[i].item()
        if seq_len > 1:
            # Valid times for this sequence
            valid_times = times_unnorm[i, :seq_len]
            intervals = valid_times[1:] - valid_times[:-1]
            time_intervals_unnorm_padded[i, :seq_len-1] = intervals
    
    # Log-normalize intervals
    # Padded values (and the last element of any sequence, and single-element sequences) will be log(epsilon)
    # if they remain 0, or log(actual_interval + epsilon)
    log_time_intervals_padded = torch.log(time_intervals_unnorm_padded.clamp(min=0) + epsilon) 
    
    # Ensure padded regions are definitely zero in log-space (or a consistent padding like log(epsilon))
    # The above clamp and add epsilon handles non-positive intervals before log.
    # For actual padding tokens beyond seq_len, they should be masked or ignored by encoder.
    # Here, we ensure that positions that are padding OR beyond seq_len-1 (for intervals) are set.
    # Create a mask for interval padding
    interval_padding_mask = torch.ones_like(attention_mask) # True means pad
    for i in range(B):
        seq_len = original_lengths[i].item()
        if seq_len > 1:
            interval_padding_mask[i, :seq_len-1] = False # These are valid interval positions
        # All other positions remain True (to be padded)
    
    log_time_intervals_padded.masked_fill_(interval_padding_mask, torch.log(torch.tensor(epsilon, device=device))) # or 0.0 if preferred for padding

    return log_time_intervals_padded