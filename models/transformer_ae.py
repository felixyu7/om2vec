import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

import time

from utils import nll_poisson_loss # Relative import

class TransformerAE(pl.LightningModule): # Renamed class
    """Transformer Autoencoder (AE) / Variational Autoencoder (VAE).
    
    Can operate as a standard AE or a VAE depending on the `is_vae` flag.
    Supports different beta schedules for the KL divergence term in VAE mode.
    """
    def __init__(self,
                 in_features: int = 6400,
                 latent_dim: int = 64,
                 embed_dim: int = 32,
                 is_vae: bool = False, # Flag to enable VAE specific parts
                 beta_schedule: str = 'none', # 'none', 'constant', 'annealed'
                 beta_factor: float = 1e-5,
                 beta_peak_epoch: int = 4, # Only used if beta_schedule='annealed'
                 sensor_positional_encoding: bool = False,
                 dataset_size: int = -1,
                 batch_size: int = 128,
                 lr: float = 1e-3, 
                 lr_schedule: list[int] | None = None,
                 weight_decay: float = 1e-5):
        super().__init__()
        # Save hyperparameters, including new ones for VAE control
        self.save_hyperparameters(
            'in_features', 'latent_dim', 'embed_dim', 'is_vae', 'beta_schedule',
            'beta_factor', 'beta_peak_epoch', 'sensor_positional_encoding', 
            'dataset_size', 'batch_size', 'lr', 'lr_schedule', 'weight_decay'
        )

        # VAE requires twice the latent dim output from encoder core
        encoder_output_dim = latent_dim * 2 if is_vae else latent_dim
        
        self.embedding = nn.Linear(1, embed_dim)
        self.initial_downsample = nn.Sequential(nn.Linear(in_features, 1024),
                                                nn.LeakyReLU())
        # Encoder TCN Blocks (structure remains the same)
        self.encoder_blocks = nn.Sequential(Transformer_Enc_Block(dim=embed_dim,
                                                          num_heads=4,
                                                          seq_len_in=1024, 
                                                          seq_len_out=512),
                                       Transformer_Enc_Block(dim=embed_dim,
                                                          num_heads=4,
                                                          seq_len_in=512,
                                                          seq_len_out=256),
                                       Transformer_Enc_Block(dim=embed_dim,
                                                          num_heads=4,
                                                          seq_len_in=256, 
                                                          seq_len_out=encoder_output_dim) # Adjust output size
                                       )
        self.latent_output = nn.Linear(embed_dim, 1)

        # Latent projection layers (only fc_mu needed if AE)
        self.fc_latent = nn.Linear(encoder_output_dim, latent_dim) # For AE: z = fc_latent(h)
        if is_vae:
            # For VAE: mu = fc_latent(h_mu), logvar = fc_logvar(h_logvar)
            # Assumes encoder output h has size latent_dim * 2
            # We reuse fc_latent for mu, add fc_logvar
            self.fc_logvar = nn.Linear(latent_dim, latent_dim) # VAE needs separate logvar head

        # --- Decoder --- (Structure remains the same)
        self.decoder_embedding = nn.Linear(1, embed_dim)
        self.mem_embedding = nn.Parameter(torch.randn(latent_dim, embed_dim))
        self.decoder_blocks = nn.Sequential(Transformer_Dec_Block(dim=embed_dim,
                                                        num_heads=4,
                                                        seq_len_in=latent_dim, 
                                                        seq_len_out=256),
                                    Transformer_Dec_Block(dim=embed_dim,
                                                        num_heads=4,
                                                        seq_len_in=256, 
                                                        seq_len_out=512),
                                     Transformer_Dec_Block(dim=embed_dim,
                                                        num_heads=4,
                                                        seq_len_in=512, 
                                                        seq_len_out=1024),
                                     Transformer_Dec_Block(dim=embed_dim,
                                                        num_heads=4,
                                                        seq_len_in=1024, 
                                                        seq_len_out=in_features)
                                     )
        self.output = nn.Linear(embed_dim, 1)
        
        # Positional encoding layer (conditional creation)
        self.pos_encoder = nn.Linear(3, in_features) if sensor_positional_encoding else None
        
        # Beta calculation setup
        self.beta = 0.0
        self.iter = 0
        # Ensure total_steps is positive if needed for annealing
        self.total_steps = max(0, dataset_size * beta_peak_epoch) if dataset_size > 0 else 0

    def _encode_core(self, inputs, pos=None):
        # Internal method to get the raw output from the encoder backbone
        if self.hparams.sensor_positional_encoding and pos is not None and self.pos_encoder is not None:
            inputs = inputs + self.pos_encoder(pos)
        inputs = self.initial_downsample(inputs)
        h = self.embedding(inputs.unsqueeze(-1))
        h = self.encoder_blocks(h)
        h = self.latent_output(h).squeeze(-1) # Shape: (B, encoder_output_dim)
        return h

    def encode(self, inputs, pos=None): # Public encode method
        """Encodes input into latent representation (z for AE, mu for VAE)."""
        h = self._encode_core(inputs, pos)
        if self.hparams.is_vae:
             # Split encoder output for VAE: h -> h_mu, h_logvar
             h_mu, h_logvar = torch.chunk(h, 2, dim=1)
             mu = self.fc_latent(h_mu) # Use fc_latent for mu
             return mu # Return mu for VAE latent representation
        else:
             z = self.fc_latent(h) # Use fc_latent for AE latent representation
             return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Decoder remains the same, takes latent vector z (shape: B, latent_dim)
        z_emb = self.decoder_embedding(z.unsqueeze(-1))
        mem = self.mem_embedding.unsqueeze(0).expand(z_emb.size(0), -1, -1)
        decoded_h = z_emb
        for layer in self.decoder_blocks:
            decoded_h = layer(decoded_h, mem)
        outputs = self.output(decoded_h)
        return outputs.squeeze(-1)
    
    def forward(self, inputs, pos=None):
        h = self._encode_core(inputs, pos)
        
        if self.hparams.is_vae:
            h_mu, h_logvar = torch.chunk(h, 2, dim=1)
            mu = self.fc_latent(h_mu)
            logvar = self.fc_logvar(h_logvar) # Get logvar using its dedicated layer
            z = self.reparameterize(mu, logvar)
            outputs = self.decode(z)
            return outputs, mu, logvar # Return all for VAE loss calculation
        else:
            z = self.fc_latent(h) # AE latent vector
            outputs = self.decode(z)
            return outputs, z # Return z for AE (no mu/logvar)

    def kl_divergence(self, mu, logvar):
        # Sum over latent dim, result shape (B,)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def _calculate_beta(self):
         # Calculate beta based on schedule
         if not self.hparams.is_vae or self.hparams.beta_schedule == 'none':
              return 0.0
         elif self.hparams.beta_schedule == 'constant':
              return self.hparams.beta_factor
         elif self.hparams.beta_schedule == 'annealed':
              if self.total_steps > 0:
                   # Cosine annealing
                   current_beta = self.hparams.beta_factor * \
                                  ((np.cos(np.pi * (self.iter / self.total_steps - 1)) + 1) / 2)
                   return current_beta
              else:
                   return self.hparams.beta_factor # Fallback if total_steps is 0
         else:
              print(f"Warning: Unknown beta_schedule '{self.hparams.beta_schedule}'. Defaulting to 0.")
              return 0.0

    def _common_step(self, batch, batch_idx, stage):
        # Handle input batch structure
        if self.hparams.sensor_positional_encoding:
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                 raise ValueError("Batch should be a tuple/list of (inputs, pos) when sensor_positional_encoding is True.")
            inputs, pos = batch
        else:
            pos = None
            if isinstance(batch, (list, tuple)):
                 inputs = batch[0] 
            else:
                 inputs = batch

        # Forward pass
        forward_output = self(inputs, pos)
        outputs = forward_output[0]

        # Calculate reconstruction loss
        # Apply softmax before loss, consistent with previous implementation
        reconstruction_loss = nll_poisson_loss(inputs, torch.softmax(outputs, dim=-1))
        
        total_loss = reconstruction_loss
        kl_loss = torch.tensor(0.0, device=self.device) # Default KL loss is 0

        # VAE specific calculations
        if self.hparams.is_vae:
            if len(forward_output) != 3:
                 raise ValueError("VAE forward pass should return outputs, mu, logvar.")
            mu, logvar = forward_output[1], forward_output[2]
            kl_loss = self.kl_divergence(mu, logvar).mean() # Average KL over batch
            
            # Calculate beta for VAE loss
            beta = self._calculate_beta()
            if stage == 'train':
                 self.iter += 1 # Increment training iteration counter only in training
            
            total_loss = reconstruction_loss + (beta * kl_loss)
            self.log(f"{stage}_beta", beta, batch_size=self.hparams.batch_size, sync_dist=False)
        
        # Logging
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

# --- Helper Modules (Renamed internal classes for clarity) ---

class Transformer_Enc_Block(nn.Module): # Renamed from Transformer_VAE_Enc
    def __init__(self, dim=256,
                 num_heads=8,
                 ff_dim=2048,
                 dropout=0.1,
                 seq_len_in=5000,
                 seq_len_out=1000):
        super().__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                    nhead=num_heads, 
                                                    dim_feedforward=ff_dim, 
                                                    dropout=dropout,
                                                    activation=F.gelu, 
                                                    batch_first=True,
                                                    norm_first=True)
        self.norm = nn.LayerNorm(dim)
        self.downsample = nn.Linear(seq_len_in, seq_len_out)
        
    def forward(self, x):
        x = self.enc_layer(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1) 
        x = self.downsample(x)
        x = x.permute(0, 2, 1)
        return x

class Transformer_Dec_Block(nn.Module): # Renamed from Transformer_VAE_Dec
    def __init__(self, dim=256,
                 num_heads=8,
                 ff_dim=2048,
                 dropout=0.1,
                 seq_len_in=1000,
                 seq_len_out=5000):
        super().__init__()
        self.dec_layer = nn.TransformerDecoderLayer(d_model=dim, 
                                                    nhead=num_heads, 
                                                    dim_feedforward=ff_dim, 
                                                    dropout=dropout,
                                                    activation=F.gelu,
                                                    batch_first=True,
                                                    norm_first=True)
        self.norm = nn.LayerNorm(dim)
        self.upsample = nn.Linear(seq_len_in, seq_len_out)

    def forward(self, x, memory):
        x = self.dec_layer(tgt=x, memory=memory) 
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        return x 