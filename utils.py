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