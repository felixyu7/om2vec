import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import zuko
from utils import PositionalEncoding

class NT_VAE(pl.LightningModule):
    def __init__(self,
                 # in_features removed
                 latent_dim=128,
                 embed_dim=32, # Used for Transformer encoder d_model
                 beta_factor=1e-5,
                 beta_peak_epoch=4,
                 sensor_positional_encoding=True, # Handling of this will be reviewed, currently ignored in forward
                 max_seq_len_padding=512,
                 transformer_encoder_layers=6,
                 transformer_encoder_heads=8,
                 transformer_encoder_ff_dim=2048, # d_model * 4 is common
                 transformer_encoder_dropout=0.1,
                 flow_transforms=5,
                 flow_bins=8,
                 flow_hidden_dim=128, # Example, ensure it's a list if Zuko expects list e.g. [128, 128]
                 flow_hidden_layers=2, # Used to construct hidden_features list for Zuko
                 batch_size=32, # Added batch_size to hparams for logging
                 lr=1e-3,
                 lr_schedule=[2, 20],
                 weight_decay=1e-5,
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder: PyTorch Transformer based
        self.encoder_input_embedding = nn.Linear(2, self.hparams.embed_dim) # Input: (time, count)
        self.pos_encoder = PositionalEncoding(self.hparams.embed_dim, 
                                              self.hparams.transformer_encoder_dropout, 
                                              max_len=self.hparams.max_seq_len_padding)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.transformer_encoder_heads,
            dim_feedforward=self.hparams.transformer_encoder_ff_dim,
            dropout=self.hparams.transformer_encoder_dropout,
            batch_first=True # Important: input will be (B, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.transformer_encoder_layers
        )
        self.to_latent_mu = nn.Linear(self.hparams.embed_dim, self.hparams.latent_dim)
        self.to_latent_logvar = nn.Linear(self.hparams.embed_dim, self.hparams.latent_dim)
        
        # Decoder: Conditional Neural Spline Flow (Zuko)
        self.conditional_flow = zuko.flows.NSF(
            features=1, # Modeling univariate distribution of raw_times_padded at each step
            context=self.hparams.latent_dim,
            transforms=self.hparams.flow_transforms,
            bins=self.hparams.flow_bins,
            hidden_features=[self.hparams.flow_hidden_dim] * self.hparams.flow_hidden_layers,
        )
        
        if self.hparams.sensor_positional_encoding:
            # self.sensor_positional_encoding = nn.Linear(3, self.hparams.in_features) # Defunct, in_features removed
            # Sensor positional encoding is currently ignored in the forward pass.
            # If re-enabled, its initialization and application need to be revisited.
            pass
        
        self.beta = 0.
        self.current_train_iter = 0 # Renamed from self.iter for clarity
        
        self.test_step_results = {'num_hits': [], 'js_divs': []}

    def encode(self, times_data, counts_data, attention_mask):
        # times_data, counts_data: (B, S)
        # attention_mask: (B, S), boolean, True for valid tokens (False for padding)
        
        concatenated_input = torch.stack((times_data, counts_data), dim=-1).float() # (B, S, 2)
        embedded_input = self.encoder_input_embedding(concatenated_input) # (B, S, embed_dim)
        embedded_input = self.pos_encoder(embedded_input) # Add positional encoding (B,S,E)
        
        # PyTorch TransformerEncoder expects src_key_padding_mask where True means PADDED/MASKED
        # Current attention_mask is True for VALID tokens. So, invert it.
        src_key_padding_mask = ~attention_mask # (B, S)
        
        encoded_sequence = self.transformer_encoder(embedded_input, src_key_padding_mask=src_key_padding_mask) # (B, S, embed_dim)
        
        # Masked average pooling over sequence dimension
        # Ensure attention_mask is float for multiplication, and keepdim for division
        float_attention_mask = attention_mask.unsqueeze(-1).float()
        masked_encoded_sequence = encoded_sequence * float_attention_mask
        
        summed_pool = masked_encoded_sequence.sum(dim=1) # (B, embed_dim)
        # Number of valid tokens for each sequence in the batch
        num_valid_tokens = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1) # (B, 1), avoid division by zero
        
        pooled_output = summed_pool / num_valid_tokens # (B, embed_dim)
        
        mu = self.to_latent_mu(pooled_output)
        logvar = self.to_latent_logvar(pooled_output)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_log_prob(self, target_log_times, z, attention_mask):
        # target_log_times: (B, S), e.g., batch['times_padded'] (log-transformed)
        # z: (B, latent_dim)
        # attention_mask: (B, S), boolean, True for valid tokens

        B, S = target_log_times.shape
        target_flat = target_log_times.reshape(-1, 1).float() # (B*S, 1)

        context_expanded = z.unsqueeze(1).expand(-1, S, -1) # (B, S, latent_dim)
        context_flat = context_expanded.reshape(-1, self.hparams.latent_dim) # (B*S, latent_dim)

        # distribution object from flow conditioned on context
        dist = self.conditional_flow(context_flat)
        log_probs_flat = dist.log_prob(target_flat) # (B*S,)
        log_probs_seq = log_probs_flat.reshape(B, S) # (B, S)

        masked_log_probs = log_probs_seq * attention_mask.float() # Apply mask, ensure float for multiplication
        return masked_log_probs
    
    def decode(self, z, time_steps_raw):
        """
        Decodes a latent representation z to produce PDF values for photon arrival times
        at the given raw time_steps.
        The internal flow models log(t+1). This method transforms the PDF back to raw time scale.

        Args:
            z (torch.Tensor): Latent representation, shape (B, latent_dim).
            time_steps_raw (torch.Tensor): Time steps at which to evaluate the PDF, in raw scale.
                                           Can be shape (B, NumTimeSteps) or (NumTimeSteps).
        Returns:
            torch.Tensor: PDF values at the given time_steps_raw, shape (B, NumTimeSteps).
        """
        B = z.shape[0]
        
        # Ensure time_steps_raw is on the same device as z
        time_steps_raw = time_steps_raw.to(z.device)

        if time_steps_raw.ndim == 1:
            time_steps_raw = time_steps_raw.unsqueeze(0).expand(B, -1) # (B, NumTimeSteps)
        
        NumTimeSteps = time_steps_raw.shape[1]

        # Transform raw time steps to log scale for the flow: log(t_raw + 1)
        # Ensure input to log is positive. Add a small epsilon if times can be zero.
        # Given that these are arrival times, they should be > 0. Adding 1 already handles t_raw=0.
        time_steps_log = torch.log(time_steps_raw.float() + 1.0)
        
        time_steps_log_flat = time_steps_log.reshape(-1, 1) # (B*NumTimeSteps, 1)

        context_expanded = z.unsqueeze(1).expand(-1, NumTimeSteps, -1) # (B, NumTimeSteps, latent_dim)
        context_flat = context_expanded.reshape(-1, self.hparams.latent_dim) # (B*NumTimeSteps, latent_dim)

        dist = self.conditional_flow(context_flat)
        
        # Log PDF in the log-transformed space
        log_pdf_log_scale_flat = dist.log_prob(time_steps_log_flat) # (B*NumTimeSteps,)
        
        # PDF in the log-transformed space
        pdf_log_scale_flat = torch.exp(log_pdf_log_scale_flat)

        # Jacobian of the transformation y = log(x+1) is dy/dx = 1/(x+1)
        # pdf_raw(x) = pdf_log(log(x+1)) * |d(log(x+1))/dx|
        # pdf_raw(x) = pdf_log(log(x+1)) * (1 / (x + 1))
        # x corresponds to time_steps_raw
        jacobian_flat = (1.0 / (time_steps_raw.float().reshape(-1) + 1.0)).clamp(min=1e-9) # (B*NumTimeSteps,)
        
        pdf_raw_flat = pdf_log_scale_flat * jacobian_flat # (B*NumTimeSteps,)
        
        pdf_values_raw = pdf_raw_flat.reshape(B, NumTimeSteps)
        
        return pdf_values_raw
    
    def forward(self, batch):
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        mu, logvar = self.encode(times_padded, counts_padded, attention_mask)
        z = self.reparameterize(mu, logvar)
        # The old decode and softmax are removed. Log prob calculation is in training_step.
        return mu, logvar, z

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def training_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
            
        # Reconstruction Loss from NSF (modeling log-transformed times)
        target_log_times = batch['times_padded'].float() # Use log-transformed times
        attention_mask = batch['attention_mask'].bool() # ensure boolean

        log_probs_masked = self.decode_log_prob(target_log_times, z, attention_mask)
        
        # Average NLL over valid (non-padded) elements in the batch
        reconstruction_loss = -log_probs_masked.sum() / attention_mask.sum().float().clamp(min=1)

        kl_loss = self.kl_divergence(mu, logvar)
        
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        # cosine annealing for beta term
        # Calculate total_steps_for_beta_annealing on the first training step or if not set
        if not hasattr(self, 'total_steps_for_beta_annealing') or self.total_steps_for_beta_annealing == 0:
            if self.trainer and hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader:
                steps_per_epoch = len(self.trainer.train_dataloader)
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch
            else: # Fallback if trainer info not available yet (shouldn't happen in training_step)
                self.total_steps_for_beta_annealing = 1 # Avoid division by zero, beta will ramp fast

        if self.total_steps_for_beta_annealing > 0:
            # Ensure current_train_iter does not exceed total_steps_for_beta_annealing for the cosine term
            # to prevent beta from decreasing after reaching peak.
            progress_ratio = min(self.current_train_iter / self.total_steps_for_beta_annealing, 1.0)
            self.beta = self.hparams.beta_factor * ((np.cos(np.pi * (progress_ratio - 1)) + 1) / 2)
        else:
            self.beta = self.hparams.beta_factor # Or some other default/fixed beta if total_steps is 0
        
        self.current_train_iter += 1
        
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        target_log_times = batch['times_padded'].float() # Use log-transformed times
        attention_mask = batch['attention_mask'].bool()

        log_probs_masked = self.decode_log_prob(target_log_times, z, attention_mask)
        reconstruction_loss = -log_probs_masked.sum() / attention_mask.sum().float().clamp(min=1)
        
        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss) # Consider if beta should be fixed for val/test
        
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True) # Renamed val_train_loss to val_loss
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        target_log_times = batch['times_padded'].float() # Use log-transformed times
        attention_mask = batch['attention_mask'].bool()

        log_probs_masked = self.decode_log_prob(target_log_times, z, attention_mask)
        reconstruction_loss = -log_probs_masked.sum() / attention_mask.sum().float().clamp(min=1)

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss) # Consider if beta should be fixed for val/test
        
        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]