import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from utils import PositionalEncoding, calculate_summary_stats, convert_absolute_times_to_log_intervals, calculate_sequence_lengths

class NT_VAE(pl.LightningModule):
    def __init__(self,
                 latent_dim=128,
                 embed_dim=32,
                 beta_factor=1e-5,
                 beta_peak_epoch=4,
                 max_seq_len_padding=512,
                 transformer_encoder_layers=6,
                 transformer_encoder_heads=8,
                 transformer_encoder_ff_dim=2048,
                 transformer_encoder_dropout=0.1,
                 transformer_decoder_layers=4,
                 transformer_decoder_heads=8,
                 transformer_decoder_ff_dim=256,
                 transformer_decoder_dropout=0.1,
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[10, 1e-6],
                 weight_decay=1e-5,
                 charge_loss_weight=1.0,
                 interval_loss_weight=1.0
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Summary stats dimensions - stored directly in latent vector z
        self.summary_stats_dim = 4
        self.z_content_dim = self.hparams.latent_dim - self.summary_stats_dim
        

        # Encoder: PyTorch Transformer based with CLS token
        # Input combines q_sequence (1D) and time_input_features (1D) -> 2D input
        self.encoder_input_embedding = nn.Linear(2, self.hparams.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hparams.embed_dim) * 0.02)
        self.pos_encoder = PositionalEncoding(self.hparams.embed_dim,
                                              self.hparams.transformer_encoder_dropout,
                                              max_len=self.hparams.max_seq_len_padding + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.transformer_encoder_heads,
            dim_feedforward=self.hparams.transformer_encoder_ff_dim,
            activation='gelu',
            dropout=self.hparams.transformer_encoder_dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.transformer_encoder_layers
        )

        # Encoder projection to content latent space only (not full latent_dim)
        self.encoder_to_latent_input_dim = self.hparams.embed_dim + 3
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.z_content_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.z_content_dim)

        # --- Transformer Encoder based Decoder ---
        # Decoder takes full latent_dim (summary_stats + content)
        self.decoder_latent_to_input_proj = nn.Linear(self.hparams.latent_dim, self.hparams.embed_dim)
        self.decoder_pos_encoder = PositionalEncoding(
            self.hparams.embed_dim,
            self.hparams.transformer_decoder_dropout,
            max_len=self.hparams.max_seq_len_padding
        )
        # Learnable content embeddings for each decoder position
        self.query_embed = nn.Parameter(
            torch.randn(self.hparams.max_seq_len_padding, self.hparams.embed_dim) * 0.02
        )

        decoder_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.transformer_decoder_heads,
            dim_feedforward=self.hparams.transformer_decoder_ff_dim,
            activation='gelu',
            dropout=self.hparams.transformer_decoder_dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_encoder_layer,
            num_layers=self.hparams.transformer_decoder_layers
        )

        # Decoder outputs: raw scores for softmax partitioning (no activation)
        self.output_projection_charge = nn.Linear(self.hparams.embed_dim, 1)
        self.output_projection_intervals = nn.Linear(self.hparams.embed_dim, 1)

        self.beta = 0.

    def encode(self, charges_log_norm_padded, times_log_norm_abs_padded, sensor_pos_padded, attention_mask):
        """
        Encode input sequences to latent space parameters.
        Summary statistics and time intervals for encoder input are calculated internally.
        
        Args:
            charges_log_norm_padded: (B, S) - log1p normalized charges.
            times_log_norm_abs_padded: (B, S) - log normalized ABSOLUTE times.
            sensor_pos_padded: (B, 3) - sensor position.
            attention_mask: (B, S) - True for padding tokens.
        
        Returns:
            mu_full: (B, latent_dim) - full mu including summary stats.
            logvar_full: (B, latent_dim) - full logvar including summary stats.
            stats_dict: Dictionary containing calculated summary statistics and original_lengths.
                        Used by forward pass for loss calculation targets.
            encoder_time_features_padded: (B,S) - log_normalized time intervals for loss calculation.
        """
        B, S = charges_log_norm_padded.shape
        device = charges_log_norm_padded.device

        # 1. Calculate summary statistics using utility function
        # times_log_norm_abs_padded contains log(absolute_times + epsilon)
        stats_dict = calculate_summary_stats(
            charges_log_norm_padded,
            times_log_norm_abs_padded,
            attention_mask
        )
        log_seq_length = stats_dict['log_seq_length']
        log_total_charge = stats_dict['log_total_charge']
        log_first_hit_time = stats_dict['log_first_hit_time'] # This is log_t_first_original
        log_last_hit_time = stats_dict['log_last_hit_time']
        original_lengths = stats_dict['original_lengths'] # Needed for interval calculation next

        # 2. Convert absolute times to log-normalized intervals for encoder input
        encoder_time_features_padded = convert_absolute_times_to_log_intervals(
            times_log_norm_abs_padded,
            original_lengths, # Use original_lengths from stats_dict
            attention_mask
        )

        # 3. Form summary stats tensor for direct inclusion in latent vector
        summary_stats_tensor = torch.stack([
            log_seq_length,
            log_total_charge,
            log_first_hit_time,
            log_last_hit_time
        ], dim=-1)  # (B, 4)

        # 4. Embed input sequences (combine charges and ENCODER time features - i.e., intervals)
        # charges_log_norm_padded is (B,S), encoder_time_features_padded is (B,S)
        concatenated_input = torch.stack((charges_log_norm_padded, encoder_time_features_padded), dim=-1).float() # (B, S, 2)
        embedded_input = self.encoder_input_embedding(concatenated_input)  # (B, S, embed_dim)

        # 5. Prepend CLS token to each sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        embedded_with_cls = torch.cat([cls_tokens, embedded_input], dim=1)  # (B, 1+S, embed_dim)

        # 6. Apply positional encoding
        embedded_with_cls = self.pos_encoder(embedded_with_cls)  # (B, 1+S, embed_dim)

        # 7. Prepare attention mask for CLS token + sequence
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B, 1)
        # The original attention_mask is for the sequence of length S
        extended_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 1+S)

        # 8. Pass through transformer encoder
        encoded_sequence = self.transformer_encoder(
            embedded_with_cls,
            src_key_padding_mask=extended_attention_mask
        )  # (B, 1+S, embed_dim)

        # 9. Extract CLS token representation (first token)
        cls_representation = encoded_sequence[:, 0, :]  # (B, embed_dim)

        # 10. Concatenate CLS representation with sensor position
        encoder_latent_input = torch.cat((cls_representation, sensor_pos_padded), dim=1)  # (B, embed_dim + 3)

        # 11. Project to content latent parameters only
        mu_content = self.to_latent_mu(encoder_latent_input)  # (B, z_content_dim)
        logvar_content = self.to_latent_logvar(encoder_latent_input)  # (B, z_content_dim)

        # Clamp logvar for stability
        logvar_content = torch.clamp(logvar_content, min=-10, max=10)

        # 12. Construct full mu and logvar by concatenating summary stats
        mu_full = torch.cat([summary_stats_tensor, mu_content], dim=-1)  # (B, latent_dim)
        
        # Summary stats have very small (near-zero) variance to make them deterministic
        deterministic_logvar = torch.full_like(summary_stats_tensor, -10.0)  # Very small but not underflowing variance
        logvar_full = torch.cat([deterministic_logvar, logvar_content], dim=-1)  # (B, latent_dim)

        return mu_full, logvar_full, stats_dict, encoder_time_features_padded

    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, inference=False):
        """
        Decode latent vector z to reconstructed sequences using softmax partitioning.
        
        Args:
            z: (B, latent_dim) - full latent vector including summary stats
            inference: (bool) - If True, returns absolute (t,q) pairs.
                               If False, returns (reconstructed_charges, reconstructed_intervals) for training.
            
        Returns:
            If inference is False:
                reconstructed_charges: (B, max_len) - reconstructed charge sequence (scaled)
                reconstructed_intervals: (B, max_len) - reconstructed interval sequence (scaled)
            If inference is True:
                List of Tensors: Each tensor is (seq_len, 2) with [time, charge] pairs.
        """
        B = z.size(0)
        device = z.device

        # Split z into summary stats and content
        summary_stats_from_z = z[:, :self.summary_stats_dim]  # (B, 4)
        # z_content = z[:, self.summary_stats_dim:] # Not directly used in this decoder version after projection

        # Extract individual summary stats
        log_seq_length_z = summary_stats_from_z[:, 0]
        log_total_charge_z = summary_stats_from_z[:, 1]
        log_t_first_original_z = summary_stats_from_z[:, 2]
        log_last_hit_time_z = summary_stats_from_z[:, 3]

        # Determine actual sequence lengths for reconstruction
        # Fix: Use differentiable values during training, floor during inference
        if inference:
            actual_seq_lengths = torch.floor(torch.exp(log_seq_length_z)).long().clamp(
                min=1, max=self.hparams.max_seq_len_padding
            )
        else:
            # During training, keep differentiable but use round for integer operations
            seq_lengths_continuous = torch.exp(log_seq_length_z).clamp(
                min=1.0, max=float(self.hparams.max_seq_len_padding)
            )
            actual_seq_lengths = torch.round(seq_lengths_continuous).long()

        # 1. Project full latent z and expand to sequence length for Transformer input
        z_projected = self.decoder_latent_to_input_proj(z)  # (B, embed_dim)
        
        # Expand z_projected and add learnable positional content embeddings
        z_projected_expanded = z_projected.unsqueeze(1).expand(-1, self.hparams.max_seq_len_padding, -1)
        
        # Add learnable content embeddings
        decoder_input_sequence = z_projected_expanded + self.query_embed.unsqueeze(0)  # (B, max_len, embed_dim)
        
        # 2. Add sinusoidal positional encoding
        decoder_input_sequence = self.decoder_pos_encoder(decoder_input_sequence)  # (B, max_len, embed_dim)

        # 3. Create masks for valid positions based on actual sequence lengths (used by Transformer and later)
        range_tensor = torch.arange(self.hparams.max_seq_len_padding, device=device).unsqueeze(0).expand(B, -1)
        actual_seq_lengths_expanded = actual_seq_lengths.unsqueeze(1).expand(-1, self.hparams.max_seq_len_padding)
        valid_mask = range_tensor < actual_seq_lengths_expanded  # (B, max_len). True for valid, False for padding.

        # 4. Pass through Transformer Encoder (acting as decoder)
        # Pass ~valid_mask as src_key_padding_mask (True for padding positions)
        transformed_sequence = self.decoder_transformer(
            src=decoder_input_sequence,
            src_key_padding_mask=~valid_mask
        )  # (B, max_len, embed_dim)

        # 5. Project to raw scores (logits) for softmax partitioning
        raw_charge_scores = self.output_projection_charge(transformed_sequence).squeeze(-1)  # (B, max_len)
        raw_interval_scores = self.output_projection_intervals(transformed_sequence).squeeze(-1)  # (B, max_len)

        # 6. Softmax partitioning for charges
        masked_charge_scores = raw_charge_scores.clone()
        masked_charge_scores[~valid_mask] = -float('inf')  # Mask invalid positions
        
        charge_probs = torch.softmax(masked_charge_scores, dim=-1)  # (B, max_len)
        
        # Reconstruct total charge and partition it
        actual_total_charge = torch.exp(log_total_charge_z) - 1.0  # Inverse of log1p
        # actual_total_charge is always non-negative due to upstream logic
        
        reconstructed_charges = charge_probs * actual_total_charge.unsqueeze(-1)  # (B, max_len)

        # 7. L-1 Mask Interval Partitioning for guaranteed last hit time
        # For each sequence of length L, there are exactly L-1 intervals between hits
        # We partition the total duration over exactly these L-1 positions
        
        # Create interval mask: True for first (L-1) positions, False elsewhere
        interval_mask = torch.zeros_like(valid_mask, dtype=torch.bool)  # (B, max_len)
        
        # Vectorized mask creation
        seq_len_minus_one = (actual_seq_lengths - 1).clamp(min=0)  # (B,)
        range_tensor = torch.arange(self.hparams.max_seq_len_padding, device=device).unsqueeze(0).expand(B, -1)  # (B, max_len)
        interval_mask = (range_tensor < seq_len_minus_one.unsqueeze(1)) & (actual_seq_lengths.unsqueeze(1) > 1)
        
        # Apply masking to interval scores
        masked_interval_scores = raw_interval_scores.clone()
        masked_interval_scores[~interval_mask] = -float('inf')  # Mask positions outside L-1 range
        
        # Handle L=1 sequences (no intervals to partition)
        all_inf_rows = torch.all(masked_interval_scores == -float('inf'), dim=-1)
        interval_probs = torch.zeros_like(masked_interval_scores)
        
        # Only apply softmax to sequences with L > 1 (which have valid intervals)
        if not torch.all(all_inf_rows):
            valid_rows = ~all_inf_rows  # Sequences with at least one interval to partition
            softmax_result = torch.softmax(masked_interval_scores[valid_rows], dim=-1)
            interval_probs[valid_rows] = softmax_result.to(interval_probs.dtype)
        # Sequences with L=1 remain with all-zero interval probabilities
        
        # Calculate total duration and partition it
        actual_t_first = torch.exp(log_t_first_original_z)
        actual_t_last = torch.exp(log_last_hit_time_z)
        total_duration = (actual_t_last - actual_t_first).clamp(min=1e-9) # Add epsilon for stability
        
        reconstructed_intervals = interval_probs * total_duration.unsqueeze(-1)  # (B, max_len)

        if not inference:
            return {
                'reconstructed_charges': reconstructed_charges,
                'reconstructed_intervals': reconstructed_intervals
            }
        else:
            output_tq_pairs = []
            for i in range(B):
                seq_len = actual_seq_lengths[i].item()
                # seq_len == 0 should never occur due to clamp(min=1)

                current_t_first = actual_t_first[i]
                current_charges = reconstructed_charges[i, :seq_len]
                
                if seq_len == 1:
                    absolute_times = current_t_first.unsqueeze(0)
                else:
                    # Intervals are for seq_len-1 elements
                    current_intervals = reconstructed_intervals[i, :seq_len-1]
                    cumulative_intervals = torch.cumsum(current_intervals, dim=0)
                    absolute_times = torch.cat((current_t_first.unsqueeze(0), current_t_first + cumulative_intervals))
                
                # Ensure absolute_times has seq_len elements
                # absolute_times.shape[0] != seq_len should never occur with current logic


                tq_pair = torch.stack((absolute_times, current_charges), dim=-1)
                output_tq_pairs.append(tq_pair)
            return output_tq_pairs

    def forward(self, batch):
        """Forward pass through the VAE."""
        # Extract inputs from batch (dataloader now provides log-norm charges and log-norm ABSOLUTE times)
        charges_log_norm_padded = batch['charges_log_norm_padded'].float()
        times_log_norm_abs_padded = batch['times_log_norm_padded'].float() # These are absolute times
        attention_mask = batch['attention_mask'].bool()  # True for padding
        sensor_pos_padded = batch['sensor_pos_batched'].float()

        # Encode - this now calculates summary stats and encoder_time_features (intervals) internally
        mu, logvar, stats_dict_from_encode, encoder_time_features_target = self.encode(
            charges_log_norm_padded=charges_log_norm_padded,
            times_log_norm_abs_padded=times_log_norm_abs_padded, # Pass absolute times
            sensor_pos_padded=sensor_pos_padded,
            attention_mask=attention_mask
        )
        
        original_lengths = stats_dict_from_encode['original_lengths'] # Get original_lengths from encode output

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        # When training/evaluating losses, inference is False
        decoded_output = self.decode(z=z, inference=False)
        reconstructed_charges = decoded_output['reconstructed_charges']
        reconstructed_intervals = decoded_output['reconstructed_intervals']
        
        # For loss calculation:
        # 'original_charges' are the input log-normalized charges (unpadded by loss function)
        # 'original_intervals' are the log-normalized intervals that the encoder would have used as input
        # (these are also unpadded by the loss function).
        # The 'encoder_time_features_target' from encode output are these target intervals.

        return {
            'mu': mu,
            'logvar': logvar,
            'reconstructed_charges': reconstructed_charges,
            'reconstructed_intervals': reconstructed_intervals,
            'original_charges_log_norm_padded': charges_log_norm_padded, # Target for charge recon
            'original_intervals_log_norm_padded': encoder_time_features_target, # Target for interval recon
            'original_lengths': original_lengths # For unpadding targets in loss
        }

    def _calculate_loss(self, forward_output):
        """Calculate reconstruction and KL divergence losses."""
        mu_full = forward_output['mu']
        logvar_full = forward_output['logvar']
        reconstructed_charges = forward_output['reconstructed_charges']
        reconstructed_intervals = forward_output['reconstructed_intervals']
        # These are the targets for reconstruction, already log-normalized and padded.
        # The loss function will handle unpadding based on original_lengths.
        original_charges_log_norm_padded = forward_output['original_charges_log_norm_padded']
        original_intervals_log_norm_padded = forward_output['original_intervals_log_norm_padded']
        
        original_lengths = forward_output['original_lengths']

        device = mu_full.device
        B = mu_full.size(0)

        # Split mu and logvar to get content parts only for KL divergence
        mu_content = mu_full[:, self.summary_stats_dim:]  # (B, z_content_dim)
        logvar_content = logvar_full[:, self.summary_stats_dim:]  # (B, z_content_dim)

        # KL divergence on content part only
        kld_loss = -0.5 * torch.sum(1 + logvar_content - mu_content.pow(2) - logvar_content.exp(), dim=1)
        kld_loss = kld_loss.mean()

        # Vectorized reconstruction losses with proper masking
        # Create masks for valid positions
        max_len = reconstructed_charges.size(1)
        
        # Ensure original targets match decoder output size
        original_charges_size = original_charges_log_norm_padded.size(1)
        original_intervals_size = original_intervals_log_norm_padded.size(1)
        
        # Pad original charges to match decoder output size
        if original_charges_size < max_len:
            # Pad with zeros
            padding_size = max_len - original_charges_size
            original_charges_log_norm_padded = torch.cat([
                original_charges_log_norm_padded,
                torch.zeros(B, padding_size, device=device, dtype=original_charges_log_norm_padded.dtype)
            ], dim=1)

        # Pad original intervals to match decoder output size
        if original_intervals_size < max_len:
            # Pad with zeros
            padding_size = max_len - original_intervals_size
            original_intervals_log_norm_padded = torch.cat([
                original_intervals_log_norm_padded,
                torch.zeros(B, padding_size, device=device, dtype=original_intervals_log_norm_padded.dtype)
            ], dim=1)
        
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        charge_valid_mask = range_tensor < original_lengths.unsqueeze(1)
        
        # Charge reconstruction loss (vectorized)
        # Fix: Convert linear reconstructed values to log-normalized to match targets
        pred_charges_log_norm = torch.log1p(reconstructed_charges)
        
        # Compute smooth L1 loss and mask invalid positions
        charge_losses = F.smooth_l1_loss(pred_charges_log_norm, original_charges_log_norm_padded, reduction='none')
        charge_losses = charge_losses * charge_valid_mask.float()
        
        # Average over valid positions only
        valid_charge_count = charge_valid_mask.sum()
        charge_recon_loss = charge_losses.sum() / (valid_charge_count + 1e-8)

        # Interval reconstruction loss (vectorized)
        # Create mask for valid interval positions (seq_len > 1 and within interval range)
        interval_valid_mask = charge_valid_mask.clone()
        # For intervals, we need seq_len-1 valid positions, so shift the mask
        interval_valid_mask = interval_valid_mask[:, :-1]  # Remove last column
        # Also mask out sequences with length <= 1
        sequences_with_intervals = (original_lengths > 1).unsqueeze(1)
        interval_valid_mask = interval_valid_mask & sequences_with_intervals
        
        if interval_valid_mask.any():
            # Fix: Convert linear reconstructed intervals to log-normalized to match targets
            pred_intervals_log_norm = torch.log(reconstructed_intervals[:, :-1].clamp(min=0) + 1e-9)
            target_intervals = original_intervals_log_norm_padded[:, :-1]
            
            # Compute smooth L1 loss and mask invalid positions
            interval_losses = F.smooth_l1_loss(pred_intervals_log_norm, target_intervals, reduction='none')
            interval_losses = interval_losses * interval_valid_mask.float()
            
            # Average over valid positions only
            valid_interval_count = interval_valid_mask.sum()
            interval_recon_loss = interval_losses.sum() / (valid_interval_count + 1e-8)
        else:
            interval_recon_loss = torch.tensor(0.0, device=device)

        return {
            'kld_loss': kld_loss,
            'charge_recon_loss': charge_recon_loss,
            'interval_recon_loss': interval_recon_loss
        }

    def training_step(self, batch, batch_idx):
        forward_output = self(batch)
        losses = self._calculate_loss(forward_output)

        # Beta annealing - Fix: Use global_step from trainer for checkpoint compatibility
        steps_per_epoch = len(self.trainer.train_dataloader)
        self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch

        # Fix: Use trainer's global_step instead of manual counter for checkpoint compatibility
        current_step = self.trainer.global_step
        
        if self.total_steps_for_beta_annealing > 0:
            progress_ratio = min(current_step / self.total_steps_for_beta_annealing, 1.0)
            self.beta = self.hparams.beta_factor * ((np.cos(np.pi * (progress_ratio - 1)) + 1) / 2)
        else:
            self.beta = self.hparams.beta_factor

        # Total loss
        total_loss = (self.beta * losses['kld_loss'] +
                     self.hparams.charge_loss_weight * losses['charge_recon_loss'] +
                     self.hparams.interval_loss_weight * losses['interval_recon_loss'])

        # Keep manual counter as backup
        # self.current_train_iter is no longer used

        # Logging
        self.log("train_loss", total_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("train_kl_loss", losses['kld_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_charge_recon_loss", losses['charge_recon_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_interval_recon_loss", losses['interval_recon_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        forward_output = self(batch)
        losses = self._calculate_loss(forward_output)

        # Total loss
        total_loss = (self.beta * losses['kld_loss'] + 
                     self.hparams.charge_loss_weight * losses['charge_recon_loss'] +
                     self.hparams.interval_loss_weight * losses['interval_recon_loss'])

        # Logging
        self.log("val_loss", total_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", losses['kld_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_charge_recon_loss", losses['charge_recon_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_interval_recon_loss", losses['interval_recon_loss'], batch_size=self.hparams.batch_size, sync_dist=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        forward_output = self(batch)
        losses = self._calculate_loss(forward_output)

        # Total loss
        total_loss = (self.beta * losses['kld_loss'] + 
                     self.hparams.charge_loss_weight * losses['charge_recon_loss'] +
                     self.hparams.interval_loss_weight * losses['interval_recon_loss'])

        # Logging
        self.log("test_loss", total_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("test_kl_loss", losses['kld_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_charge_recon_loss", losses['charge_recon_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_interval_recon_loss", losses['interval_recon_loss'], batch_size=self.hparams.batch_size, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # Fix: Use trainer's max_epochs for T_max to avoid premature decay
        if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs:
            T_max = self.trainer.max_epochs
        else:
            T_max = self.hparams.lr_schedule[0]  # Fallback
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=self.hparams.lr_schedule[1]
        )
        return [optimizer], [scheduler]