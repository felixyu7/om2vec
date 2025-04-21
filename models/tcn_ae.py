import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import lightning.pytorch as pl
import numpy as np

from ..utils import nll_poisson_loss # Using Poisson NLL for consistency

# ────────────────────────────
#  Causal‑TCN building blocks (Moved from tcn_ae.py)
# ────────────────────────────
class CausalConv1d(nn.Conv1d):
    """Padding only on the left so y[t] sees x[:t]."""
    def __init__(self, in_ch, out_ch, k, dilation=1, **kw):
        pad = (k - 1) * dilation
        super().__init__(in_ch, out_ch, k,
                         padding=pad,
                         dilation=dilation,
                         **kw)

    def forward(self, x):                       # x: (B,C,T)
        y = super().forward(x)
        return y[..., : x.size(-1)]             # trim right‑side padding

class TemporalBlock(nn.Module):
    """(CausalConv → ReLU → Dropout) × 2 + residual."""
    def __init__(self, in_ch, out_ch, k, dil, p_drop):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(CausalConv1d(in_ch, out_ch, k, dil)),
            nn.ReLU(),
            nn.Dropout(p_drop),
            weight_norm(CausalConv1d(out_ch, out_ch, k, dil)),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return F.relu(self.net(x) + self.downsample(x))

class TCN(nn.Module):
    """
    Stacked TemporalBlocks with exponentially growing dilation.
    Example: channels=[64,64,64,64] ⇒ 4 layers.
    """
    def __init__(self,
                 in_ch: int,
                 channels: list[int],
                 k: int = 3,
                 p_drop: float = 0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dil = 2 ** i
            inp_ch = in_ch if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(inp_ch, out_ch, k, dil, p_drop)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x: (B,C_in,T)
        return self.net(x)                      # (B,C_out,T)

# ────────────────────────────
#  TCN Auto‑encoder (AE/VAE compatible)
# ────────────────────────────
class TCN1DAutoencoder(nn.Module):
    """
    Autoregressive 1‑D auto‑encoder (or VAE) for fixed‑length sequences.
    Uses TCN blocks for encoding and autoregressive decoding.
    Set `is_vae=True` to enable VAE functionality.
    """
    def __init__(self,
                 seq_len: int,
                 latent_dim: int = 32,
                 is_vae: bool = False, # Added flag
                 channels: list[int] | None = None,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.is_vae = is_vae
        channels = channels or [64, 64, 64, 64] 

        # --- Encoder --- 
        self.encoder_tcn = TCN(1, channels, kernel_size, dropout)
        self.enc_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer size depends on whether it's VAE or AE
        encoder_output_dim = latent_dim * 2 if is_vae else latent_dim
        self.to_latent_proj = nn.Linear(channels[-1], encoder_output_dim)

        # --- Decoder (Autoregressive) --- 
        self.dec_input_embed = nn.Conv1d(1, channels[0], 1) 
        self.latent_to_dec_init = nn.Linear(latent_dim, channels[0]) # Decoder always uses latent_dim
        self.decoder_tcn = TCN(channels[0], channels, kernel_size, dropout)
        self.out_proj = nn.Conv1d(channels[-1], 1, 1) 

    def encode(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """(B,T) → (B,D_latent) for AE or ((B, D_latent), (B, D_latent)) for VAE (mu, logvar)."""
        h = self.encoder_tcn(x.unsqueeze(1)) # (B, C_last, T)
        h = self.enc_pool(h).squeeze(-1)     # (B, C_last)
        h = self.to_latent_proj(h)           # (B, encoder_output_dim)
        
        if self.is_vae:
            mu, logvar = torch.chunk(h, 2, dim=1) # Split into mu and logvar
            return mu, logvar
        else:
            z = h # Use directly as latent vector z for AE
            return z

    def decode(self,
               z: torch.Tensor,                      # (B,D_latent)
               gt: torch.Tensor | None = None,       # (B,T) ground truth for teacher forcing
               teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """Autoregressive decoding loop."""
        B = z.size(0)
        T = self.seq_len
        device = z.device
        dec_init_state = self.latent_to_dec_init(z).unsqueeze(-1) 
        prev_output = torch.zeros((B, 1), device=device)
        outputs = []
        for t in range(T):
            step_input = prev_output.unsqueeze(-1) 
            h = self.dec_input_embed(step_input) + dec_init_state 
            h = self.decoder_tcn(h) 
            step_output = self.out_proj(h)
            current_output = step_output.squeeze(-1)
            outputs.append(current_output)
            use_teacher_force = (self.training and gt is not None and 
                                 torch.rand(1, device=device) < teacher_forcing_ratio)
            if use_teacher_force:
                prev_output = gt[:, t].unsqueeze(1) 
            else:
                prev_output = current_output 
        return torch.cat(outputs, dim=1)

    def forward(self, 
                x: torch.Tensor,                      # (B,T) Input sequence
                teacher_forcing_ratio: float = 0.0):  # Ratio for training
        """(B,T) → (recon: (B,T), z/mu: (B,D_latent), [logvar: (B,D_latent)])"""
        encoded_output = self.encode(x)
        
        if self.is_vae:
            mu, logvar = encoded_output
            # Need to reparameterize to get z for the decoder
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon = self.decode(z, gt=x, teacher_forcing_ratio=teacher_forcing_ratio)
            return recon, mu, logvar # Return all VAE components
        else:
            z = encoded_output
            recon = self.decode(z, gt=x, teacher_forcing_ratio=teacher_forcing_ratio)
            return recon, z # Return AE components

# ────────────────────────────
#  Lightning Wrapper for TCN AE/VAE (Autoregressive)
# ────────────────────────────
class TCNAE(pl.LightningModule): # Renamed class
    """PyTorch Lightning wrapper for the Autoregressive TCN Autoencoder/VAE.
    
    Uses teacher forcing during training. Includes optional VAE components.
    """
    def __init__(self,
                 seq_len: int = 6400,
                 latent_dim: int = 64,
                 tcn_channels: list[int] | None = None,
                 tcn_kernel_size: int = 3,
                 tcn_dropout: float = 0.1,
                 is_vae: bool = False, # Added flag
                 beta_schedule: str = 'none', # 'none', 'constant', 'annealed'
                 beta_factor: float = 1e-5,
                 beta_peak_epoch: int = 4, # Only used if beta_schedule='annealed' & is_vae
                 teacher_forcing_ratio: float = 0.5,
                 dataset_size: int = -1, 
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 lr_schedule: list[int] | None = None,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters(
            'seq_len', 'latent_dim', 'tcn_channels', 'tcn_kernel_size',
            'tcn_dropout', 'is_vae', 'beta_schedule', 'beta_factor', 
            'beta_peak_epoch', 'teacher_forcing_ratio', 'dataset_size', 
            'batch_size', 'lr', 'lr_schedule', 'weight_decay'
        )

        self.model = TCN1DAutoencoder(
            seq_len=self.hparams.seq_len,
            latent_dim=self.hparams.latent_dim,
            is_vae=self.hparams.is_vae, # Pass flag down
            channels=self.hparams.tcn_channels,
            kernel_size=self.hparams.tcn_kernel_size,
            dropout=self.hparams.tcn_dropout
        )

        # Beta calculation setup
        self.beta = 0.0
        self.iter = 0
        self.total_steps = max(0, dataset_size * beta_peak_epoch) if dataset_size > 0 else 0

    # Public encode method for inference/conversion
    def encode(self, inputs, pos=None): # pos unused
        """Encodes input into latent representation (z for AE, mu for VAE)."""
        encoded_output = self.model.encode(inputs)
        if self.hparams.is_vae:
            mu, _ = encoded_output # Return only mu for VAE
            return mu
        else:
            z = encoded_output # Return z for AE
            return z

    # Forward pass delegates to internal model
    def forward(self, inputs, pos=None, teacher_forcing_ratio=0.0): # pos unused
        return self.model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def _calculate_beta(self):
         # Shared beta calculation logic
         if not self.hparams.is_vae or self.hparams.beta_schedule == 'none': return 0.0
         if self.hparams.beta_schedule == 'constant': return self.hparams.beta_factor
         if self.hparams.beta_schedule == 'annealed':
              if self.total_steps > 0:
                   return self.hparams.beta_factor * ((np.cos(np.pi * (self.iter / self.total_steps - 1)) + 1) / 2)
              else: return self.hparams.beta_factor # Fallback
         print(f"Warning: Unknown beta_schedule '{self.hparams.beta_schedule}'. Defaulting to 0."); return 0.0

    def _common_step(self, batch, batch_idx, stage):
        if isinstance(batch, (list, tuple)):
            inputs = batch[0] 
        else:
            inputs = batch
        
        current_tf_ratio = self.hparams.teacher_forcing_ratio if stage == 'train' else 0.0
        forward_output = self(inputs, teacher_forcing_ratio=current_tf_ratio)
        
        outputs = forward_output[0]
        reconstruction_loss = nll_poisson_loss(inputs, torch.softmax(outputs, dim=-1))
        
        total_loss = reconstruction_loss
        kl_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.is_vae:
            if len(forward_output) != 3:
                 raise ValueError("VAE forward pass should return outputs, mu, logvar.")
            mu, logvar = forward_output[1], forward_output[2]
            kl_loss = self.kl_divergence(mu, logvar).mean()
            
            beta = self._calculate_beta()
            if stage == 'train':
                 self.iter += 1
            
            total_loss = reconstruction_loss + (beta * kl_loss)
            self.log(f"{stage}_beta", beta, batch_size=self.hparams.batch_size, sync_dist=False)

        self.log(f"{stage}_loss", total_loss, batch_size=self.hparams.batch_size, sync_dist=True if stage != 'train' else False)
        self.log(f"{stage}_reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True if stage != 'train' else False)
        if self.hparams.is_vae:
             self.log(f"{stage}_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True if stage != 'train' else False)
        if stage == 'train':
             self.log("teacher_forcing_ratio", current_tf_ratio, batch_size=self.hparams.batch_size)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val') 

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=0.1)
            return [optimizer], [scheduler]
        else:
            return optimizer