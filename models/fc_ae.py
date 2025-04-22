import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from utils import nll_poisson_loss

class FCAE(pl.LightningModule):
    """Fully Connected Autoencoder (AE) / Variational Autoencoder (VAE).
    
    Can operate as a standard AE or a VAE depending on the `is_vae` flag.
    Supports different beta schedules for the KL divergence term in VAE mode.
    """
    def __init__(self,
                 in_features: int = 6400,
                 latent_dim: int = 64,
                 fc_hidden_dims: list[int] = [1024, 512],
                 is_vae: bool = False, # Flag to enable VAE specific parts
                 beta_schedule: str = 'none', # 'none', 'constant', 'annealed'
                 beta_factor: float = 1e-5,
                 beta_peak_epoch: int = 4, # Only used if beta_schedule='annealed' & is_vae
                 dataset_size: int = -1, # Needed for annealing schedule
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 lr_schedule: list[int] | None = None,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters(
            'in_features', 'latent_dim', 'fc_hidden_dims', 'is_vae', 'beta_schedule',
            'beta_factor', 'beta_peak_epoch', 'dataset_size', 'batch_size', 'lr', 
            'lr_schedule', 'weight_decay'
        )

        # --- Encoder --- 
        encoder_layers = []
        last_dim = in_features
        for hidden_dim in fc_hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, hidden_dim))
            encoder_layers.append(nn.ReLU()) 
            last_dim = hidden_dim
        self.encoder_core = nn.Sequential(*encoder_layers)

        # Latent layers
        self.fc_latent = nn.Linear(last_dim, latent_dim) # Used for z (AE) or mu (VAE)
        if is_vae:
            self.fc_logvar = nn.Linear(last_dim, latent_dim) # Separate layer for logvar (VAE)

        # --- Decoder --- 
        decoder_layers = []
        last_dim = latent_dim 
        for hidden_dim in reversed(fc_hidden_dims):
             decoder_layers.append(nn.Linear(last_dim, hidden_dim))
             decoder_layers.append(nn.ReLU())
             last_dim = hidden_dim
        decoder_layers.append(nn.Linear(last_dim, in_features))
        self.decoder = nn.Sequential(*decoder_layers)

        # Beta calculation setup
        self.beta = 0.0
        self.iter = 0
        self.total_steps = max(0, dataset_size * beta_peak_epoch) if dataset_size > 0 else 0

    def _encode_core(self, inputs):
        # Internal method for core encoder output
        return self.encoder_core(inputs)

    def encode(self, inputs, pos=None): # Public encode method (pos unused)
        """Encodes input into latent representation (z for AE, mu for VAE)."""
        h = self._encode_core(inputs)
        if self.hparams.is_vae:
             mu = self.fc_latent(h) # Use fc_latent for mu
             return mu
        else:
             z = self.fc_latent(h) # Use fc_latent for z
             return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        outputs = self.decoder(z)
        return torch.softmax(outputs, dim=-1) # Apply softmax for Poisson NLL

    def forward(self, inputs, pos=None): # pos is unused
        h = self._encode_core(inputs)
        
        if self.hparams.is_vae:
            mu = self.fc_latent(h)
            logvar = self.fc_logvar(h) # VAE needs logvar
            z = self.reparameterize(mu, logvar)
            outputs = self.decode(z)
            return outputs, mu, logvar
        else:
            z = self.fc_latent(h) # AE gets z directly
            outputs = self.decode(z)
            return outputs, z

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def _calculate_beta(self):
         # Shared beta calculation logic (identical to TransformerAE)
         if not self.hparams.is_vae or self.hparams.beta_schedule == 'none':
              return 0.0
         elif self.hparams.beta_schedule == 'constant':
              return self.hparams.beta_factor
         elif self.hparams.beta_schedule == 'annealed':
              if self.total_steps > 0:
                   current_beta = self.hparams.beta_factor * \
                                  ((np.cos(np.pi * (self.iter / self.total_steps - 1)) + 1) / 2)
                   return current_beta
              else:
                   return self.hparams.beta_factor # Fallback
         else:
              print(f"Warning: Unknown beta_schedule '{self.hparams.beta_schedule}'. Defaulting to 0.")
              return 0.0

    def _common_step(self, batch, batch_idx, stage):
        # Ignore pos if present 
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch
        
        forward_output = self(inputs)
        outputs = forward_output[0]

        reconstruction_loss = nll_poisson_loss(inputs, outputs)
        
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