import torch
import torch.nn.functional as F
import numpy as np

def log_transform(x, eps=1e-6):
    """
    Applies a log transformation: log(x + eps).
    Handles tensor inputs.
    If x contains zeros, eps prevents log(0).
    """
    return torch.log(x + eps)

def nll_exponential(rate, target_times, mask):
    """
    Computes the negative log-likelihood for an Exponential distribution.
    Assumes rate is the parameter (lambda) of the Exponential distribution.
    PDF(t; rate) = rate * exp(-rate * t)
    log PDF(t; rate) = log(rate) - rate * t

    Args:
        rate (Tensor): The rate parameter (lambda) of the Exponential distribution. Shape (B, S, 1) or (B, S).
        target_times (Tensor): The observed inter-event times. Shape (B, S, 1) or (B, S).
        mask (Tensor): Boolean tensor indicating valid (True) and padded (False) entries. Shape (B, S).

    Returns:
        Tensor: Scalar NLL loss.
    """
    if rate.ndim > mask.ndim: # If rate is (B,S,1) and mask is (B,S)
        mask = mask.unsqueeze(-1) # Make mask (B,S,1)
    
    # Ensure rate is positive (it should be due to softplus in decoder)
    # Clamp for numerical stability if necessary, though softplus should handle it.
    # rate = torch.clamp(rate, min=eps)

    # Ensure target_times has the same number of dimensions as rate for multiplication
    if rate.ndim > target_times.ndim and rate.shape[-1] == 1 and target_times.ndim == rate.ndim -1 :
        target_times_expanded = target_times.unsqueeze(-1)
    else:
        target_times_expanded = target_times
    
    log_pdf = torch.log(rate) - rate * target_times_expanded
    
    # Apply mask
    masked_log_pdf = log_pdf * mask
    
    # Sum over sequence and batch, then normalize by number of valid entries
    loss = -torch.sum(masked_log_pdf) / torch.sum(mask).clamp(min=1.0) # Avoid division by zero
    return loss

def nll_gaussian(mean, log_std, target_values, mask):
    """
    Computes the negative log-likelihood for a Gaussian distribution.
    log PDF(x; mu, sigma) = -0.5 * log(2*pi*sigma^2) - ( (x-mu)^2 / (2*sigma^2) )

    Args:
        mean (Tensor): The mean of the Gaussian distribution. Shape (B, S, 1) or (B, S).
        log_std (Tensor): The log of the standard deviation. Shape (B, S, 1) or (B, S).
        target_values (Tensor): The observed values (e.g., charges). Shape (B, S, 1) or (B, S).
        mask (Tensor): Boolean tensor indicating valid (True) and padded (False) entries. Shape (B, S).

    Returns:
        Tensor: Scalar NLL loss.
    """
    if mean.ndim > mask.ndim: # If params are (B,S,1) and mask is (B,S)
        mask = mask.unsqueeze(-1)
    
    # Ensure target_values has the same number of dimensions as mean/log_std for subtraction
    if mean.ndim > target_values.ndim and mean.shape[-1] == 1 and target_values.ndim == mean.ndim -1:
        target_values_expanded = target_values.unsqueeze(-1)
    else:
        target_values_expanded = target_values

    std = torch.exp(log_std)
    # log(2*pi*sigma^2) = log(2*pi) + 2*log(sigma) = log(2*pi) + 2*log_std
    log_var_term = np.log(2 * np.pi) + 2 * log_std
    squared_error_term = ((target_values_expanded - mean)**2) / (2 * std**2)
    
    log_pdf = -0.5 * (log_var_term + squared_error_term)
    
    masked_log_pdf = log_pdf * mask
    
    loss = -torch.sum(masked_log_pdf) / torch.sum(mask).clamp(min=1.0)
    return loss

def kl_divergence_gaussian(mu, log_var):
    """
    Computes the KL divergence between a diagonal Gaussian q(z|x) and a standard Gaussian prior p(z) = N(0, I).
    KL(q || p) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    The sum is over the latent dimensions. We then average over the batch.

    Args:
        mu (Tensor): Mean of the q distribution. Shape (batch_size, latent_dim).
        log_var (Tensor): Log variance of the q distribution. Shape (batch_size, latent_dim).

    Returns:
        Tensor: Scalar KL divergence loss.
    """
    # Sum over latent dimensions (dim=1), then mean over batch (dim=0)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return torch.mean(kl_div)