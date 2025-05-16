import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model) if batch_first
        # If batch_first, permute to (seq_len, batch_size, d_model)
        is_batch_first = x.size(1) == self.pe.size(1) and x.size(0) != self.pe.size(0)
        if is_batch_first: # Assuming x is (B, S, E)
            x = x.permute(1, 0, 2) # (S, B, E)
        
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        
        if is_batch_first: # Permute back if originally batch_first
            x = x.permute(1, 0, 2) # (B, S, E)
        return x