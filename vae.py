import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from utils import MonotonicNN, PositionalEncoding, calculate_summary_stats, convert_absolute_times_to_log_intervals, reparameterize, calculate_mmd_loss, wasserstein_1d, wasserstein_1d_from_cdf

class NT_VAE(pl.LightningModule):
    def __init__(self,
                 latent_dim=128,
                 embed_dim=32,
                 alpha=1.0,
                 lambda_=1.0,
                 max_seq_len_padding=512,
                 transformer_encoder_layers=6,
                 transformer_encoder_heads=8,
                 transformer_encoder_ff_dim=2048,
                 transformer_encoder_dropout=0.1,
                 transformer_decoder_layers=4,
                 transformer_decoder_heads=8,
                 transformer_decoder_ff_dim=256,
                 transformer_decoder_dropout=0.1,
                 monotonic_nn_hidden_dim=128,
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[10, 1e-6],
                 weight_decay=1e-5,
                 charge_wasserstein_loss_weight=1.0,
                 interval_wasserstein_loss_weight=1.0
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.summary_stats_dim = 4
        self.z_content_dim = self.hparams.latent_dim - self.summary_stats_dim

        # Encoder
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

        self.encoder_to_latent_input_dim = self.hparams.embed_dim + 3
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.z_content_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.z_content_dim)

        # Decoder
        self.decoder_latent_to_input_proj = nn.Linear(self.hparams.latent_dim, self.hparams.embed_dim)
        self.decoder_pos_encoder = PositionalEncoding(
            self.hparams.embed_dim,
            self.hparams.transformer_decoder_dropout,
            max_len=self.hparams.max_seq_len_padding
        )
        self.query_embed = nn.Parameter(torch.randn(self.hparams.max_seq_len_padding, self.hparams.embed_dim) * 0.02)

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

        self.output_projection_charge = MonotonicNN(self.hparams.embed_dim, self.hparams.monotonic_nn_hidden_dim, 1)
        self.output_projection_intervals = MonotonicNN(self.hparams.embed_dim, self.hparams.monotonic_nn_hidden_dim, 1)


    def encode(self, charges_log_norm_padded, times_log_norm_abs_padded, sensor_pos_padded, attention_mask):
        """Encodes input sequences into latent space parameters."""
        B, S = charges_log_norm_padded.shape
        device = charges_log_norm_padded.device

        # 1. Calculate summary statistics (always needed)
        stats_dict = calculate_summary_stats(charges_log_norm_padded, times_log_norm_abs_padded, attention_mask)
        original_lengths = stats_dict['original_lengths']
        summary_stats_tensor = torch.stack([
            stats_dict['log_seq_length'],
            stats_dict['log_total_charge'],
            stats_dict['log_first_hit_time'],
            stats_dict['log_last_hit_time']
        ], dim=-1)

        # 2. Identify multi-hit sequences that require transformer processing
        is_multi_hit = original_lengths > 1
        num_multi_hit = is_multi_hit.sum()

        # Initialize content latent parameters to a deterministic zero state
        mu_content = torch.zeros(B, self.z_content_dim, device=device)
        # Initialize logvar to a large negative value for a near-zero std dev
        logvar_content = torch.full((B, self.z_content_dim), -10.0, device=device)

        # 3. Process multi-hit sequences through the transformer encoder
        if num_multi_hit > 0:
            # Select only the multi-hit sequences for processing
            mh_charges = charges_log_norm_padded[is_multi_hit]
            mh_times_abs = times_log_norm_abs_padded[is_multi_hit]
            mh_lengths = original_lengths[is_multi_hit]
            mh_attn_mask = attention_mask[is_multi_hit]
            mh_sensor_pos = sensor_pos_padded[is_multi_hit]

            # Convert absolute times to log-normalized intervals
            mh_time_features = convert_absolute_times_to_log_intervals(mh_times_abs, mh_lengths, mh_attn_mask)
            
            # Embed input sequences
            mh_concatenated_input = torch.stack((mh_charges, mh_time_features), dim=-1)
            mh_embedded_input = self.encoder_input_embedding(mh_concatenated_input)

            # Prepend CLS token
            mh_cls_tokens = self.cls_token.expand(num_multi_hit, -1, -1)
            mh_embedded_with_cls = torch.cat([mh_cls_tokens, mh_embedded_input], dim=1)
            mh_embedded_with_cls = self.pos_encoder(mh_embedded_with_cls)

            # Prepare attention mask
            mh_cls_mask = torch.zeros(num_multi_hit, 1, dtype=torch.bool, device=device)
            mh_extended_attention_mask = torch.cat([mh_cls_mask, mh_attn_mask], dim=1)

            # Pass through transformer encoder
            mh_encoded_sequence = self.transformer_encoder(mh_embedded_with_cls, src_key_padding_mask=mh_extended_attention_mask)
            
            # Extract CLS token representation
            mh_cls_representation = mh_encoded_sequence[:, 0, :]
            
            # Concatenate CLS representation with sensor position
            mh_encoder_latent_input = torch.cat((mh_cls_representation, mh_sensor_pos), dim=1)

            # Project to content latent parameters
            mh_mu_content = self.to_latent_mu(mh_encoder_latent_input).to(mu_content.dtype)
            mh_logvar_content = self.to_latent_logvar(mh_encoder_latent_input).clamp(min=-10, max=10).to(logvar_content.dtype)

            # Scatter results back to the full batch tensors
            mu_content[is_multi_hit] = mh_mu_content
            logvar_content[is_multi_hit] = mh_logvar_content

        # 4. For 1-hit sequences, mu_content and logvar_content remain zero (deterministic content)
        
        # 5. Construct full mu and logvar
        mu_full = torch.cat([summary_stats_tensor, mu_content], dim=-1)
        deterministic_logvar = torch.full_like(summary_stats_tensor, -10.0)
        logvar_full = torch.cat([deterministic_logvar, logvar_content], dim=-1)

        # Note: encoder_time_features_padded is needed as a target for the loss function
        encoder_time_features_padded = convert_absolute_times_to_log_intervals(times_log_norm_abs_padded, original_lengths, attention_mask)

        return mu_full, logvar_full, stats_dict, encoder_time_features_padded


    def decode(self, z, inference=False):
        """Decodes latent vector z into reconstructed sequences."""
        B, device = z.size(0), z.device
        max_len = self.hparams.max_seq_len_padding

        # 1. Unpack summary stats and determine sequence lengths
        summary_stats = z[:, :self.summary_stats_dim]
        log_seq_length_z, log_total_charge_z, log_t_first_z, log_t_last_z = summary_stats.T
        seq_lengths_continuous = torch.exp(log_seq_length_z).clamp(min=1.0, max=float(max_len))
        actual_seq_lengths = torch.floor(seq_lengths_continuous).long() if inference else torch.round(seq_lengths_continuous).long()

        # 2. Identify multi-hit sequences that require transformer processing
        is_multi_hit = actual_seq_lengths > 1
        num_multi_hit = is_multi_hit.sum()

        # Initialize full output tensors
        charge_cdfs = torch.zeros(B, max_len, device=device)
        interval_cdfs = torch.zeros(B, max_len, device=device)
        # For inference, we'll need the PDF, so we'll store it if needed.
        charge_pdfs = torch.zeros(B, max_len, device=device)
        interval_pdfs = torch.zeros(B, max_len, device=device)

        # 3. Handle 1-hit sequences deterministically
        # For a 1-hit sequence, all charge probability is in the first bin.
        # For a 1-hit sequence, the CDF is a step function at the first bin.
        charge_cdfs[~is_multi_hit, 0:] = 1.0
        charge_pdfs[~is_multi_hit, 0] = 1.0
        # Interval CDFs/PDFs remain all zero as there are no intervals.

        # 4. Process multi-hit sequences through the transformer decoder
        if num_multi_hit > 0:
            # Select only the multi-hit sequences for processing
            mh_z = z[is_multi_hit]
            mh_lengths = actual_seq_lengths[is_multi_hit]

            # Prepare input for Transformer Decoder
            mh_z_projected = self.decoder_latent_to_input_proj(mh_z)
            mh_decoder_input = mh_z_projected.unsqueeze(1) + self.query_embed.unsqueeze(0)
            mh_decoder_input = self.decoder_pos_encoder(mh_decoder_input)

            # Transformer Decoding
            mh_valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < mh_lengths.unsqueeze(1)
            mh_transformed_sequence = self.decoder_transformer(src=mh_decoder_input, src_key_padding_mask=~mh_valid_mask)

            # Reconstruct Charges via Monotonic NN
            mh_unnorm_charge_pdf = self.output_projection_charge(mh_transformed_sequence).squeeze(-1)
            mh_unnorm_charge_pdf.masked_fill_(~mh_valid_mask, 0)
            mh_charge_pdf = mh_unnorm_charge_pdf / (mh_unnorm_charge_pdf.sum(dim=-1, keepdim=True) + 1e-9)
            mh_charge_cdf = torch.cumsum(mh_charge_pdf, dim=-1)

            # Reconstruct Time Intervals via Monotonic NN
            mh_unnorm_interval_pdf = self.output_projection_intervals(mh_transformed_sequence).squeeze(-1)
            mh_interval_mask = torch.arange(max_len, device=device).unsqueeze(0) < (mh_lengths - 1).unsqueeze(1)
            mh_unnorm_interval_pdf.masked_fill_(~mh_interval_mask, 0)
            mh_interval_pdf = mh_unnorm_interval_pdf / (mh_unnorm_interval_pdf.sum(dim=-1, keepdim=True) + 1e-9)
            mh_interval_cdf = torch.cumsum(mh_interval_pdf, dim=-1)

            # Scatter results back to the full batch tensors
            charge_cdfs[is_multi_hit] = mh_charge_cdf
            interval_cdfs[is_multi_hit] = mh_interval_cdf
            charge_pdfs[is_multi_hit] = mh_charge_pdf
            interval_pdfs[is_multi_hit] = mh_interval_pdf

        # 5. Format output based on whether we are in inference mode or not
        if not inference:
            return {'charge_cdfs': charge_cdfs, 'interval_cdfs': interval_cdfs}
        else:
            # This part is for generating actual sequences for visualization or analysis
            output_tq_pairs = []
            actual_total_charge = torch.exp(log_total_charge_z) - 1.0
            reconstructed_charges = charge_pdfs * actual_total_charge.unsqueeze(-1)
            
            actual_t_first = torch.exp(log_t_first_z)
            total_duration = (torch.exp(log_t_last_z) - actual_t_first).clamp(min=1e-9)
            reconstructed_intervals = interval_pdfs * total_duration.unsqueeze(-1)

            for i in range(B):
                seq_len = actual_seq_lengths[i].item()
                t_first = actual_t_first[i]
                charges = reconstructed_charges[i, :seq_len]
                if seq_len > 1:
                    intervals = reconstructed_intervals[i, :seq_len-1]
                    absolute_times = torch.cat([t_first.unsqueeze(0), t_first + torch.cumsum(intervals, dim=0)])
                else:
                    absolute_times = t_first.unsqueeze(0)
                output_tq_pairs.append(torch.stack((absolute_times, charges), dim=-1))
            return output_tq_pairs

    def forward(self, batch):
        """Forward pass through the VAE."""
        # Ensure all input tensors are on the correct device and dtype
        charges_log_norm_padded = batch['charges_log_norm_padded'].to(self.dtype)
        times_log_norm_abs_padded = batch['times_log_norm_padded'].to(self.dtype)
        attention_mask = batch['attention_mask'].bool() # Mask should be boolean
        sensor_pos_padded = batch['sensor_pos_batched'].to(self.dtype)

        mu, logvar, stats_dict, encoder_time_features_target = self.encode(
            charges_log_norm_padded, times_log_norm_abs_padded, sensor_pos_padded, attention_mask
        )

        z = reparameterize(mu, logvar)
        z = z.to(self.dtype)  # Ensure z has the correct dtype
        decoded_output = self.decode(z, inference=False)

        return {
            'mu': mu,
            'logvar': logvar,
            'charge_cdfs': decoded_output['charge_cdfs'],
            'interval_cdfs': decoded_output['interval_cdfs'],
            'original_charges_log_norm_padded': charges_log_norm_padded,
            'original_intervals_log_norm_padded': encoder_time_features_target,
            'original_lengths': stats_dict['original_lengths']
        }

    def _calculate_loss(self, forward_output):
        """Calculate reconstruction and KL divergence losses."""
        mu_full = forward_output['mu']
        logvar_full = forward_output['logvar']
        charge_cdfs = forward_output['charge_cdfs']
        interval_cdfs = forward_output['interval_cdfs']
        original_charges_log_norm_padded = forward_output['original_charges_log_norm_padded']
        original_intervals_log_norm_padded = forward_output['original_intervals_log_norm_padded']
        original_lengths = forward_output['original_lengths']

        device = mu_full.device
        B, max_len = charge_cdfs.shape

        # KLD Loss
        mu_content = mu_full[:, self.summary_stats_dim:]
        logvar_content = logvar_full[:, self.summary_stats_dim:]
        kld_loss = 0.5 * (mu_content.pow(2) + logvar_content.exp() - 1.0 - logvar_content).sum(dim=1).mean()

        # --- Pad original targets to match decoder output size ---
        original_charges_size = original_charges_log_norm_padded.size(1)
        if original_charges_size < max_len:
            padding_size = max_len - original_charges_size
            original_charges_log_norm_padded = F.pad(original_charges_log_norm_padded, (0, padding_size))

        original_intervals_size = original_intervals_log_norm_padded.size(1)
        if original_intervals_size < max_len:
            padding_size = max_len - original_intervals_size
            original_intervals_log_norm_padded = F.pad(original_intervals_log_norm_padded, (0, padding_size))

        # --- Reconstruction Losses ---
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
        charge_valid_mask = range_tensor < original_lengths.unsqueeze(1)

        # --- Reconstruction Losses (for multi-hit sequences only) ---
        is_multi_hit = original_lengths > 1
        num_multi_hit = is_multi_hit.sum()

        if num_multi_hit > 0:
            # Select only the multi-hit sequences for reconstruction loss calculation
            mh_charge_cdfs = charge_cdfs[is_multi_hit]
            mh_original_charges = original_charges_log_norm_padded[is_multi_hit]
            mh_charge_valid_mask = charge_valid_mask[is_multi_hit]

            # Charge Reconstruction Loss (Wasserstein from CDF)
            mh_true_charges_unnorm = torch.expm1(mh_original_charges) # Inverse of log1p
            charge_recon_loss = wasserstein_1d_from_cdf(mh_charge_cdfs, mh_true_charges_unnorm, mh_charge_valid_mask)

            # Interval Reconstruction Loss (Wasserstein from CDF)
            interval_valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < (original_lengths - 1).unsqueeze(1)
            mh_interval_cdfs = interval_cdfs[is_multi_hit]
            mh_original_intervals = original_intervals_log_norm_padded[is_multi_hit]
            mh_interval_valid_mask = interval_valid_mask[is_multi_hit]
            
            if mh_interval_valid_mask.any():
                mh_true_intervals_unnorm = torch.exp(mh_original_intervals) # Inverse of log
                interval_recon_loss = wasserstein_1d_from_cdf(mh_interval_cdfs, mh_true_intervals_unnorm, mh_interval_valid_mask)
            else:
                interval_recon_loss = torch.tensor(0.0, device=device)
        else:
            # If the batch contains only 1-hit sequences, reconstruction loss is 0
            charge_recon_loss = torch.tensor(0.0, device=device)
            interval_recon_loss = torch.tensor(0.0, device=device)

        return {
            'kld_loss': kld_loss,
            'charge_recon_loss': charge_recon_loss,
            'interval_recon_loss': interval_recon_loss
        }


    def training_step(self, batch, batch_idx):
        forward_output = self(batch)
        losses = self._calculate_loss(forward_output)
        
        z = reparameterize(forward_output['mu'], forward_output['logvar'])
        mmd_loss = calculate_mmd_loss(z[:, self.summary_stats_dim:])
        
        total_loss = (self.hparams.charge_wasserstein_loss_weight * losses['charge_recon_loss'] +
                      self.hparams.interval_wasserstein_loss_weight * losses['interval_recon_loss'] +
                      (1 - self.hparams.alpha) * losses['kld_loss'] +
                      (self.hparams.lambda_ + self.hparams.alpha - 1) * mmd_loss)

        self.log_dict({
            'train_loss': total_loss,
            'train_charge_recon_loss': losses['charge_recon_loss'],
            'train_interval_recon_loss': losses['interval_recon_loss'],
            'train_mmd_loss': mmd_loss,
            'train_kld_loss': losses['kld_loss']
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        forward_output = self(batch)
        losses = self._calculate_loss(forward_output)
        
        z = reparameterize(forward_output['mu'], forward_output['logvar'])
        mmd_loss = calculate_mmd_loss(z[:, self.summary_stats_dim:])
        
        total_loss = (self.hparams.charge_wasserstein_loss_weight * losses['charge_recon_loss'] +
                      self.hparams.interval_wasserstein_loss_weight * losses['interval_recon_loss'] +
                      (1 - self.hparams.alpha) * losses['kld_loss'] +
                      (self.hparams.lambda_ + self.hparams.alpha - 1) * mmd_loss)

        self.log_dict({
            'val_loss': total_loss,
            'val_charge_recon_loss': losses['charge_recon_loss'],
            'val_interval_recon_loss': losses['interval_recon_loss'],
            'val_mmd_loss': mmd_loss,
            'val_kld_loss': losses['kld_loss']
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        forward_output = self(batch)
        losses = self._calculate_loss(forward_output)
        
        z = reparameterize(forward_output['mu'], forward_output['logvar'])
        mmd_loss = calculate_mmd_loss(z[:, self.summary_stats_dim:])
        
        total_loss = (self.hparams.charge_wasserstein_loss_weight * losses['charge_recon_loss'] +
                      self.hparams.interval_wasserstein_loss_weight * losses['interval_recon_loss'] +
                      (1 - self.hparams.alpha) * losses['kld_loss'] +
                      (self.hparams.lambda_ + self.hparams.alpha - 1) * mmd_loss)

        self.log_dict({
            'test_loss': total_loss,
            'test_charge_recon_loss': losses['charge_recon_loss'],
            'test_interval_recon_loss': losses['interval_recon_loss'],
            'test_mmd_loss': mmd_loss,
            'test_kld_loss': losses['kld_loss']
        }, on_step=False, on_epoch=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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