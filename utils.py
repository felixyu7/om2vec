import torch

# @torch.compile # Removed for broader compatibility (requires torch>=2.0)
def nll_poisson_loss(x, x_recon, reduction='mean'):
    # x: Actual binned counts (integer values)
    # x_recon: Reconstructed probability distribution (values between 0 and 1, summing to 1)
    
    x = torch.exp(x) - 1
    
    # Calculate the rate parameter lambda for each bin
    N = x.sum(dim=-1, keepdim=True)  # Total count per sample
    lambda_ = x_recon * N  # Scale probabilities by total count
    
    # Poisson log-likelihood
    log_factorial_x = torch.lgamma(x + 1)  # log(x!)
    log_likelihood = x * torch.log(lambda_ + 1e-8) - lambda_ - log_factorial_x
    
    # Negative log-likelihood
    nll = -torch.sum(log_likelihood, dim=-1)
    
    if reduction == 'none':
        return nll
    else:
        return torch.mean(nll)