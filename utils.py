import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.distributions

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

def mdn_negative_log_likelihood(target, pis, mus, sigmas, attention_mask):
    """
    Computes the negative log-likelihood for a Mixture Density Network (MDN) output.
    Args:
        target: (B, S)
        pis: (B, S, K) - mixing coefficients (softmaxed)
        mus: (B, S, K) - means
        sigmas: (B, S, K) - stddevs (positive)
        attention_mask: (B, S) - boolean, True for valid tokens
    Returns:
        Scalar negative log-likelihood averaged over valid tokens.
    """
    # Expand target to (B, S, K)
    target_exp = target.unsqueeze(-1).expand_as(mus)
    # Compute log-probabilities under each Gaussian component
    normal = torch.distributions.Normal(mus, sigmas)
    log_probs = normal.log_prob(target_exp)  # (B, S, K)
    # Combine with log mixing coefficients
    log_pis = torch.log(pis + 1e-8)  # (B, S, K), add epsilon for stability
    log_weighted = log_pis + log_probs  # (B, S, K)
    # Log-sum-exp over mixture components
    log_sum = torch.logsumexp(log_weighted, dim=-1)  # (B, S)
    # Mask out invalid tokens
    valid_log_sum = log_sum[attention_mask]
    # Negative log-likelihood, mean over valid tokens
    nll = -valid_log_sum.mean()
    return nll

def point_estimate_mse_loss(target, prediction, attention_mask):
    """
    Computes the mean squared error between target and prediction, masked by attention_mask.
    Args:
        target: (batch, seq_len, dim)
        prediction: (batch, seq_len, dim)
        attention_mask: (batch, seq_len)
    Returns:
        Scalar mean MSE over valid tokens.
    """
    mse = F.mse_loss(prediction, target, reduction='none')  # (batch, seq_len, dim)
    mse = mse.mean(-1)  # (batch, seq_len)
    mse = mse * attention_mask
    mean_mse = mse.sum() / attention_mask.sum()
    return mean_mse