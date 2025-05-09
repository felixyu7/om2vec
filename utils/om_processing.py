import torch
import numpy as np

def calculate_summary_statistics(times: torch.Tensor, charges: torch.Tensor, charge_log_offset: float = 1.0, time_log_epsilon: float = 1e-9):
    """
    Calculates 9 log-normalized summary statistics for a single OM's photon sequence.

    Args:
        times (torch.Tensor): 1D tensor of photon arrival times for a single OM.
                              Assumed to be sorted or will be sorted internally.
        charges (torch.Tensor): 1D tensor of corresponding photon charges.
        charge_log_offset (float): Offset for log-normalizing charge-based statistics (log(x + offset)).
        time_log_epsilon (float): Small epsilon for log-normalizing time-based statistics (log(x + epsilon))
                                   to prevent log(0).

    Returns:
        torch.Tensor: 1D tensor of 9 log-normalized summary statistics.
                      Order: c_total, c_500ns, c_100ns, t_first, t_last,
                             t_20_percent, t_50_percent, t_mean, t_std.
    """
    if times.numel() == 0: # Handle empty sequences
        # Return a tensor of NaNs or zeros, appropriately shaped.
        # Using NaNs might be better for downstream debugging if this case is unexpected.
        # For now, let's use a default log-normalized zero-like value.
        # log(epsilon) for time, log(offset) for charge.
        # This needs careful consideration based on how these are used.
        # For simplicity, returning log(epsilon) for all.
        return torch.full((9,), torch.log(torch.tensor(time_log_epsilon, device=times.device)), device=times.device, dtype=torch.float32)

    # Ensure tensors are float32 for calculations
    times = times.float()
    charges = charges.float()

    # Sort by time, as many statistics depend on temporal order
    sorted_indices = torch.argsort(times)
    sorted_times = times[sorted_indices]
    sorted_charges = charges[sorted_indices]

    # 1. c_total: total DOM charge
    c_total = torch.sum(sorted_charges)

    # Make times relative to the first pulse
    t_first_val = sorted_times[0]
    relative_times = sorted_times - t_first_val

    # 2. c_500ns: charge within 500 ns of the first pulse
    mask_500ns = relative_times <= 500
    c_500ns = torch.sum(sorted_charges[mask_500ns])

    # 3. c_100ns: charge within 100 ns of the first pulse
    mask_100ns = relative_times <= 100
    c_100ns = torch.sum(sorted_charges[mask_100ns])

    # 4. t_first: relative time of first pulse (is 0 by definition of relative_times)
    #    The concept doc lists "t_first: relative time of first pulse".
    #    If the input times were absolute, this would be min(times).
    #    If we are calculating relative times internally, then t_first (as a feature)
    #    could be the actual first pulse time before relativization, or 0 if all times are relative.
    #    Let's assume it's the first pulse time from the original (potentially un-relativized) sequence.
    #    For now, using the t_first_val calculated above.
    #    If the input `times` are already relative to some global event start, then t_first_val is that relative time.
    #    If they are absolute, t_first_val is absolute.
    #    The key is consistency. The plan implies per-OM processing, so t_first_val is the OM's first hit time.
    t_first_stat = t_first_val # This is the absolute time of the first pulse in this OM.

    # 5. t_last: relative time of the last pulse
    t_last_stat = relative_times[-1]

    # Cumulative charge for t_X% calculations
    cumulative_charges = torch.cumsum(sorted_charges, dim=0)
    
    # 6. t_20%: relative time at which 20% of the charge is collected
    charge_20_percent_mark = 0.20 * c_total
    # Find first index where cumulative charge exceeds this mark
    idx_20_percent = torch.searchsorted(cumulative_charges, charge_20_percent_mark)
    idx_20_percent = torch.clamp(idx_20_percent, 0, relative_times.numel() - 1) # Ensure valid index
    t_20_percent = relative_times[idx_20_percent]

    # 7. t_50%: relative time at which 50% of the charge is collected
    charge_50_percent_mark = 0.50 * c_total
    idx_50_percent = torch.searchsorted(cumulative_charges, charge_50_percent_mark)
    idx_50_percent = torch.clamp(idx_50_percent, 0, relative_times.numel() - 1)
    t_50_percent = relative_times[idx_50_percent]

    # 8. t_mean: charge-weighted mean of the relative pulse arrival times
    if c_total > 0: # Avoid division by zero if total charge is zero
        t_mean = torch.sum(relative_times * sorted_charges) / c_total
    else:
        t_mean = torch.tensor(0.0, device=times.device) # Or some other default

    # 9. t_std: charge-weighted standard deviation of the relative pulse arrival times
    if c_total > 0:
        t_std = torch.sqrt(torch.sum(sorted_charges * (relative_times - t_mean)**2) / c_total)
    else:
        t_std = torch.tensor(0.0, device=times.device)

    # Log-normalization
    # For charges: log(x + offset) -> log1p(x) if offset is 1 and x>=0. Using config offset.
    log_c_total = torch.log(c_total + charge_log_offset)
    log_c_500ns = torch.log(c_500ns + charge_log_offset)
    log_c_100ns = torch.log(c_100ns + charge_log_offset)

    # For times:
    # t_first_stat can be large and positive.
    # relative_times are >= 0.
    # log(x + epsilon) for positive values.
    # If t_first_stat can be negative (e.g. global reference time), abs might be needed.
    # Assuming t_first_stat is non-negative (absolute time or relative to an earlier event start).
    log_t_first = torch.log(t_first_stat + time_log_epsilon if t_first_stat >= 0 else torch.abs(t_first_stat) + time_log_epsilon) # Handle potential negative if it's not truly absolute
    log_t_last = torch.log(t_last_stat + time_log_epsilon) # relative, so >= 0
    log_t_20_percent = torch.log(t_20_percent + time_log_epsilon) # relative, so >= 0
    log_t_50_percent = torch.log(t_50_percent + time_log_epsilon) # relative, so >= 0
    log_t_mean = torch.log(t_mean + time_log_epsilon) # relative, so >= 0
    log_t_std = torch.log(t_std + time_log_epsilon) # std dev, so >= 0

    summary_stats = torch.stack([
        log_c_total, log_c_500ns, log_c_100ns,
        log_t_first, log_t_last, log_t_20_percent,
        log_t_50_percent, log_t_mean, log_t_std
    ])

    return summary_stats.float()


def preprocess_photon_sequence(times: torch.Tensor, charges: torch.Tensor,
                               max_photons: int,
                               tq_log_norm_offset: float):
    """
    Preprocesses a single OM's photon sequence: normalizes t and q using log(x + offset), and pads/truncates.
    The concept.md suggests "optimally combine hits" for truncation.
    For now, implementing simple truncation or padding.

    Args:
        times (torch.Tensor): 1D tensor of photon arrival times.
        charges (torch.Tensor): 1D tensor of photon charges.
        max_photons (int): The fixed length for the sequence.
        tq_log_norm_offset (float): Offset for log(x + offset) normalization for both t and q.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Padded/truncated (t,q) sequence as a (max_photons, 2) tensor,
            and a (max_photons,) boolean mask tensor (True for valid data).
    """
    if times.numel() == 0:
        processed_sequence = torch.zeros((max_photons, 2), dtype=torch.float32, device=times.device)
        mask = torch.zeros((max_photons,), dtype=torch.bool, device=times.device)
        return processed_sequence, mask

    # Normalize time: log(t + offset)
    # Ensure times are non-negative if they represent durations or relative times starting from 0.
    # If times can be negative (e.g. relative to a point within the pulse train), this needs care.
    # Assuming times are >= 0 for log normalization.
    norm_times = torch.log(times.float().clamp(min=0) + tq_log_norm_offset)

    # Normalize charge: log(q + offset)
    norm_charges = torch.log(charges.float().clamp(min=0) + tq_log_norm_offset)

    # Combine t and q
    sequence = torch.stack((norm_times, norm_charges), dim=1)
    
    num_hits = sequence.shape[0]
    processed_sequence = torch.zeros((max_photons, 2), dtype=torch.float32, device=sequence.device)
    mask = torch.zeros((max_photons,), dtype=torch.bool, device=sequence.device)

    if num_hits >= max_photons:
        # Truncate: take the first max_photons hits (after sorting by time if not already)
        # Assuming times are already sorted or should be sorted before this step if meaningful truncation is needed.
        # For simplicity, taking the first ones as they appear.
        # A better strategy might involve sorting by time first.
        # Let's assume input `times` (and thus `sequence`) should be time-sorted for meaningful truncation.
        sorted_indices = torch.argsort(times)
        sorted_sequence = sequence[sorted_indices]
        processed_sequence = sorted_sequence[:max_photons, :]
        mask[:max_photons] = True
    else:
        # Pad
        processed_sequence[:num_hits, :] = sequence
        mask[:num_hits] = True
        
    return processed_sequence, mask


def sinusoidal_positional_encoding(max_len: int, d_model: int, device=None):
    """
    Generates sinusoidal positional encodings.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Embedding dimension.
        device: PyTorch device.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (max_len, d_model).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    position = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / d_model))
    
    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe