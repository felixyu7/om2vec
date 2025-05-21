import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000):
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