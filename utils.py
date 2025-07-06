import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_sequence_lengths(attention_mask):
    """
    Calculates the actual sequence lengths from a boolean attention mask.

    Args:
        attention_mask (torch.Tensor): Boolean tensor of shape (B, S),
                                       where True indicates a padding token.

    Returns:
        torch.Tensor: Tensor of shape (B,) containing actual sequence lengths.
    """
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
    dtype = charges_log_norm_padded.dtype

    original_lengths = calculate_sequence_lengths(attention_mask)
    log_seq_length = torch.log(original_lengths.to(dtype).clamp(min=1.0)) # Clamp for log(0)

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
    last_hit_time_unnorm = torch.zeros(B, device=device, dtype=dtype)

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
    time_intervals_unnorm_padded = torch.zeros_like(times_unnorm, dtype=times_log_norm_padded.dtype)

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
    
    log_time_intervals_padded.masked_fill_(interval_padding_mask, torch.log(torch.tensor(epsilon, device=device, dtype=times_log_norm_padded.dtype))) # or 0.0 if preferred for padding

    return log_time_intervals_padded

def reparameterize(mu, logvar):
    """Standard VAE reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the RBF (Gaussian) kernel matrix between x and y:
      k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # pairwise squared distances
    d2 = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-d2 / (2 * sigma**2) + eps)

def calculate_mmd_loss(z: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Unbiased RBF-MMD^2 between q(z) (the batch z) and p(z)=N(0,I).
    """
    # sample from the prior
    prior_z = torch.randn_like(z)

    # kernel matrices
    k_zz = rbf_kernel(z,       z,       sigma)
    k_pp = rbf_kernel(prior_z, prior_z, sigma)
    k_zp = rbf_kernel(z,       prior_z, sigma)

    n = z.size(0)
    m = prior_z.size(0)

    # subtract diagonal for unbiased estimate
    # note: torch.diagonal(k_zz).sum() extracts the diagonal entries sum
    sum_kzz = (k_zz.sum() - torch.diagonal(k_zz).sum()) / (n * (n - 1))
    sum_kpp = (k_pp.sum() - torch.diagonal(k_pp).sum()) / (m * (m - 1))
    sum_kzp = k_zp.mean()  # all pairs are off-diagonal by construction

    mmd2 = sum_kzz + sum_kpp - 2 * sum_kzp
    return mmd2

def wasserstein_1d(pred_dist, true_dist, mask, bin_width=1.0):
    mask = mask.to(pred_dist.dtype)

    pred = (pred_dist * mask).clamp(min=0)
    true = (true_dist * mask).clamp(min=0)

    # renormalise so each sequence sums to 1 over valid bins
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    true = true / true.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    pred_cdf = torch.cumsum(pred, dim=-1)
    true_cdf = torch.cumsum(true, dim=-1)

    # canonical W1 (no length normalisation), scaled by bin width
    w1 = torch.sum(torch.abs(pred_cdf - true_cdf), dim=-1) * bin_width
    return w1.mean()