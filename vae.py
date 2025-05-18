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
                 flow_hidden_dim=128,
                 flow_hidden_layers=2,
                 charge_flow_transforms=3, # New: Hyperparameters for charge flow
                 charge_flow_bins=8,       # New
                 charge_flow_hidden_dim=64,# New
                 charge_flow_hidden_layers=2,# New
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[2, 20],
                 weight_decay=1e-5,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.num_summary_stats = 9 # Fixed number of summary statistics
        # The VAE learns the remaining latent dimensions
        self.learned_latent_dim = self.hparams.latent_dim - self.num_summary_stats
        if self.learned_latent_dim <= 0:
            raise ValueError(f"latent_dim ({self.hparams.latent_dim}) must be greater than num_summary_stats ({self.num_summary_stats})")

        # Encoder: PyTorch Transformer based
        self.encoder_input_embedding = nn.Linear(2, self.hparams.embed_dim) # Input: (time, count)
        self.pos_encoder = PositionalEncoding(self.hparams.embed_dim, 
                                              self.hparams.transformer_encoder_dropout, 
                                              max_len=self.hparams.max_seq_len_padding)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.transformer_encoder_heads,
            dim_feedforward=self.hparams.transformer_encoder_ff_dim,
            activation='gelu',
            dropout=self.hparams.transformer_encoder_dropout,
            batch_first=True, # Important: input will be (B, S, E)
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.transformer_encoder_layers
        )
        # Input to these layers will be pooled_output (embed_dim) + sensor_pos (3)
        self.encoder_to_latent_input_dim = self.hparams.embed_dim + 3
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.learned_latent_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.learned_latent_dim)
        
        # Decoder: Conditional Neural Spline Flow (Zuko)
        # The context for the flow is the full latent_dim (summary_stats + learned_latents)
        # self.flow_input_affine layer is removed. Standardization will be used.
        self.conditional_flow = zuko.flows.NSF(
            features=1, # Modeling univariate distribution of standardized raw_times
            context=self.hparams.latent_dim,
            transforms=self.hparams.flow_transforms,
            bins=self.hparams.flow_bins,
            hidden_features=[self.hparams.flow_hidden_dim] * self.hparams.flow_hidden_layers,
        )
        
        # Decoder: Conditional Neural Spline Flow for Charge (Zuko)
        # Context: latent_dim (from z_full) + 1 (from standardized time)
        self.conditional_charge_flow = zuko.flows.NSF(
            features=1, # Modeling univariate distribution of standardized charge
            context=self.hparams.latent_dim + 1, # z_full + standardized time
            transforms=self.hparams.charge_flow_transforms,
            bins=self.hparams.charge_flow_bins,
            hidden_features=[self.hparams.charge_flow_hidden_dim] * self.hparams.charge_flow_hidden_layers,
        )

        # Constants for charge normalization/unnormalization
        self.approx_max_log_charge = torch.tensor(np.log(1e6), dtype=torch.float32)
        self.sqrt_12 = torch.tensor(np.sqrt(12.0), dtype=torch.float32)

        self.beta = 0.
        self.current_train_iter = 0 # Renamed from self.iter for clarity
        
        # store data_mean and data_std for standardization (set in run.py)
        self.data_mean = None
        self.data_std = None
        
        self.test_step_results = {'num_hits': [], 'js_divs': []}

    def encode(self, times_data, counts_data, attention_mask, sensor_pos_batched):
        # times_data, counts_data: (B, S)
        # attention_mask: (B, S), boolean, True for valid tokens (False for padding)
        # sensor_pos_batched: (B, 3)
        
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

        # Concatenate pooled_output with sensor_pos_batched
        encoder_latent_input = torch.cat((pooled_output, sensor_pos_batched), dim=1) # (B, embed_dim + 3)
        
        mu_learned = self.to_latent_mu(encoder_latent_input)
        logvar_learned = self.to_latent_logvar(encoder_latent_input)

        return mu_learned, logvar_learned
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_log_prob(self, target_raw_times, target_raw_counts, z, attention_mask):
        # target_raw_times: (B, S), e.g., batch['raw_times_padded'] (original raw scale)
        # target_raw_counts: (B, S), e.g., batch['raw_counts_padded'] (original raw scale)
        # z: (B, latent_dim)
        # attention_mask: (B, S), boolean, True for valid tokens

        B, S = target_raw_times.shape
        target_raw_flat = target_raw_times.reshape(-1, 1).float() # (B*S, 1)

        # Retrieve standardization stats for time from datamodule (ensure they are on the correct device)
        data_mean_time = self.trainer.datamodule.data_mean.to(target_raw_flat.device) # Assuming this is time_mean
        data_std_time = self.trainer.datamodule.data_std.to(target_raw_flat.device).clamp(min=1e-6) # Assuming this is time_std

        # Standardize time: y_t = (x_t - mean_t) / std_t
        target_time_standardized_flat = (target_raw_flat - data_mean_time) / data_std_time # (B*S, 1)
        
        # Prepare context for time flow
        time_flow_context_expanded = z.unsqueeze(1).expand(-1, S, -1) # (B, S, latent_dim)
        time_flow_context_flat = time_flow_context_expanded.reshape(-1, self.hparams.latent_dim) # (B*S, latent_dim)

        # Time Flow: log p_T_raw(t_raw | z)
        dist_time = self.conditional_flow(time_flow_context_flat)
        target_time_standardized_flat_clamped = torch.clamp(target_time_standardized_flat, min=-5, max=5) # Clamp for stability
        log_prob_t_std_flat = dist_time.log_prob(target_time_standardized_flat_clamped) # log p_T_std(t_std|z)
        
        log_abs_det_jacobian_time = -torch.log(data_std_time) # log|1/std_t|
        log_prob_t_raw_flat = log_prob_t_std_flat + log_abs_det_jacobian_time # (B*S,)
        log_probs_t_raw_seq = log_prob_t_raw_flat.reshape(B, S) # (B, S)
        masked_log_probs_time = log_probs_t_raw_seq * attention_mask.float()

        # --- Charge Flow ---
        target_raw_counts_flat = target_raw_counts.reshape(-1, 1).float() # (B*S, 1)

        # Standardize charge: q_std = f(q_raw)
        # q_std = ((log(q_raw + 1) / approx_max_log_charge) - 0.5) * sqrt_12
        log_q_raw_plus_1 = torch.log(target_raw_counts_flat + 1.0) # Corrected to + 1.0 to match dataloader
        
        # Ensure constants are on the correct device
        _approx_max_log_charge = self.approx_max_log_charge.to(log_q_raw_plus_1.device)
        _sqrt_12 = self.sqrt_12.to(log_q_raw_plus_1.device)

        q_scaled = log_q_raw_plus_1 / _approx_max_log_charge
        target_charge_standardized_flat = (q_scaled - 0.5) * _sqrt_12 # (B*S, 1)
        
        # Prepare context for charge flow: z_full and standardized time
        # time_flow_context_flat is (B*S, latent_dim) from z
        # target_time_standardized_flat_clamped is (B*S, 1)
        charge_flow_context_flat = torch.cat((time_flow_context_flat, target_time_standardized_flat_clamped), dim=1) # (B*S, latent_dim + 1)

        # Charge Flow: log p_Q_raw(q_raw | z, t_std)
        dist_charge = self.conditional_charge_flow(charge_flow_context_flat)
        target_charge_standardized_flat_clamped = torch.clamp(target_charge_standardized_flat, min=-5, max=5) # Clamp for stability
        log_prob_q_std_flat = dist_charge.log_prob(target_charge_standardized_flat_clamped) # log p_Q_std(q_std | z, t_std)

        # Jacobian for charge normalization: log |d(q_std)/d(q_raw)|
        # d(q_std)/d(q_raw) = (sqrt_12 / approx_max_log_charge) * (1 / (q_raw + 1))
        # log_abs_det_jacobian_charge = log(sqrt_12) - log(approx_max_log_charge) - log(q_raw + 1)
        # The derivative of log(x+1) is 1/(x+1). So log_q_raw_plus_1 is correct here.
        log_abs_det_jacobian_charge = torch.log(_sqrt_12) - torch.log(_approx_max_log_charge) - log_q_raw_plus_1.squeeze(-1) # (B*S,)
        
        log_prob_q_raw_flat = log_prob_q_std_flat + log_abs_det_jacobian_charge # (B*S,)
        log_probs_q_raw_seq = log_prob_q_raw_flat.reshape(B, S) # (B, S)
        masked_log_probs_charge = log_probs_q_raw_seq * attention_mask.float()
        
        return masked_log_probs_time, masked_log_probs_charge
    
    def decode(self, z, time_steps_raw, num_charge_samples=1):
        """
        Decodes a latent representation z to produce:
        1. PDF values for photon arrival times at the given raw time_steps.
        2. Sampled raw charge values associated with each time_step.

        Args:
            z (torch.Tensor): Latent representation, shape (B, latent_dim).
            time_steps_raw (torch.Tensor): Time steps at which to evaluate the PDF and generate charge,
                                           in raw scale. Can be shape (B, NumTimeSteps) or (NumTimeSteps).
            num_charge_samples (int): Number of charge samples to generate per time step.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pdf_values_time_raw (torch.Tensor): PDF values for time, shape (B, NumTimeSteps).
                - generated_charges_raw (torch.Tensor): Sampled raw charges, shape (B, NumTimeSteps, num_charge_samples).
        """
        B = z.shape[0]
        time_steps_raw = time_steps_raw.to(z.device)

        if time_steps_raw.ndim == 1:
            time_steps_raw = time_steps_raw.unsqueeze(0).expand(B, -1) # (B, NumTimeSteps)
        
        NumTimeSteps = time_steps_raw.shape[1]
        
        time_steps_raw_flat = time_steps_raw.reshape(-1, 1).float() # (B*NumTimeSteps, 1)

        # Standardize time: y_t = (x_t - mean_t) / std_t
        # Ensure data_mean and data_std are on the correct device (should be handled if they are buffers/params or set in forward)
        _data_mean_time = self.data_mean.to(z.device)
        _data_std_time = self.data_std.to(z.device)
        time_steps_standardized_flat = (time_steps_raw_flat - _data_mean_time) / _data_std_time # (B*NumTimeSteps, 1)

        # --- Time PDF Calculation ---
        time_flow_context_expanded = z.unsqueeze(1).expand(-1, NumTimeSteps, -1) # (B, NumTimeSteps, latent_dim)
        time_flow_context_flat = time_flow_context_expanded.reshape(-1, self.hparams.latent_dim) # (B*NumTimeSteps, latent_dim)

        dist_time = self.conditional_flow(time_flow_context_flat)
        log_pdf_time_standardized_flat = dist_time.log_prob(time_steps_standardized_flat) # (B*NumTimeSteps,)
        pdf_time_standardized_flat = torch.exp(log_pdf_time_standardized_flat)
        
        abs_det_jacobian_time = 1.0 / _data_std_time
        pdf_time_raw_flat = pdf_time_standardized_flat * abs_det_jacobian_time
        pdf_values_time_raw = pdf_time_raw_flat.reshape(B, NumTimeSteps)

        # --- Charge Sampling and Un-normalization ---
        # Context for charge flow: z (expanded) and standardized time
        # time_flow_context_flat is (B*NumTimeSteps, latent_dim)
        # time_steps_standardized_flat is (B*NumTimeSteps, 1)
        charge_flow_context_flat = torch.cat((time_flow_context_flat, time_steps_standardized_flat), dim=1) # (B*NumTimeSteps, latent_dim + 1)
        
        dist_charge = self.conditional_charge_flow(charge_flow_context_flat)
        
        # Sample standardized charges: (B*NumTimeSteps, num_charge_samples)
        # charge_flow_context_flat is (B*NumTimeSteps, latent_dim + 1)
        
        num_elements = B * NumTimeSteps
        expanded_charge_context = charge_flow_context_flat.repeat_interleave(num_charge_samples, dim=0) # (B*NumTimeSteps*num_samples, latent_dim + 1)
        
        dist_charge_expanded = self.conditional_charge_flow(expanded_charge_context)
        sampled_charge_standardized_expanded_flat = dist_charge_expanded.sample() # (B*NumTimeSteps*num_samples, 1)
        
        # Reshape to (B*NumTimeSteps, num_charge_samples) for unnormalization
        sampled_charge_standardized = sampled_charge_standardized_expanded_flat.reshape(num_elements, num_charge_samples)

        # Un-normalize charge: q_raw = exp(((q_std / sqrt_12) + 0.5) * approx_max_log_charge) - 1
        _approx_max_log_charge_dev = self.approx_max_log_charge.to(z.device)
        _sqrt_12_dev = self.sqrt_12.to(z.device)

        q_scaled_inv = (sampled_charge_standardized / _sqrt_12_dev) + 0.5
        log_q_raw_plus_1_inv = q_scaled_inv * _approx_max_log_charge_dev
        generated_charges_raw_flat = torch.exp(log_q_raw_plus_1_inv) - 1.0
        
        # Reshape to (B, NumTimeSteps, num_charge_samples)
        generated_charges_raw = generated_charges_raw_flat.reshape(B, NumTimeSteps, num_charge_samples)
        
        return pdf_values_time_raw, generated_charges_raw
    
    def forward(self, batch):
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        summary_stats = batch['summary_stats_batched'].float() # (B, num_summary_stats)
        sensor_pos_batched = batch['sensor_pos_batched'].float() # (B, 3)

        mu_learned, logvar_learned = self.encode(times_padded, counts_padded, attention_mask, sensor_pos_batched)
        z_learned = self.reparameterize(mu_learned, logvar_learned) # (B, learned_latent_dim)
        
        # Concatenate summary statistics with learned latents
        z_full = torch.cat((summary_stats, z_learned), dim=1) # (B, latent_dim)
        
        # mu and logvar for KL loss are from the learned part only
        return mu_learned, logvar_learned, z_full

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def training_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
            
        target_raw_times = batch['raw_times_padded'].float()
        target_raw_counts = batch['raw_counts_padded'].float() # Get raw counts
        attention_mask = batch['attention_mask'].bool()

        masked_log_probs_time, masked_log_probs_charge = self.decode_log_prob(
            target_raw_times, target_raw_counts, z, attention_mask
        )
        
        # Average NLL for time
        reconstruction_loss_time = -masked_log_probs_time.sum() / attention_mask.sum().float().clamp(min=1)
        # Average NLL for charge
        reconstruction_loss_charge = -masked_log_probs_charge.sum() / attention_mask.sum().float().clamp(min=1)
        
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge # Combine losses

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
        
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        target_raw_times = batch['raw_times_padded'].float()
        target_raw_counts = batch['raw_counts_padded'].float() # Get raw counts
        attention_mask = batch['attention_mask'].bool()

        masked_log_probs_time, masked_log_probs_charge = self.decode_log_prob(
            target_raw_times, target_raw_counts, z, attention_mask
        )
        
        reconstruction_loss_time = -masked_log_probs_time.sum() / attention_mask.sum().float().clamp(min=1)
        reconstruction_loss_charge = -masked_log_probs_charge.sum() / attention_mask.sum().float().clamp(min=1)
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge
        
        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss) # Consider if beta should be fixed for val/test
        
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        target_raw_times = batch['raw_times_padded'].float()
        target_raw_counts = batch['raw_counts_padded'].float() # Get raw counts
        attention_mask = batch['attention_mask'].bool()

        masked_log_probs_time, masked_log_probs_charge = self.decode_log_prob(
            target_raw_times, target_raw_counts, z, attention_mask
        )

        reconstruction_loss_time = -masked_log_probs_time.sum() / attention_mask.sum().float().clamp(min=1)
        reconstruction_loss_charge = -masked_log_probs_charge.sum() / attention_mask.sum().float().clamp(min=1)
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss) # Consider if beta should be fixed for val/test
        
        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[0], eta_min=self.hparams.lr_schedule[1])
        return [optimizer], [scheduler]