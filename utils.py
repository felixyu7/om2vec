import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

@torch.compile
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
    
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]