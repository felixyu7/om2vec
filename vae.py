import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import PositionalEncoding, calculate_summary_stats, convert_absolute_times_to_log_intervals, reparameterize, calculate_mmd_loss

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
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[10, 1e-6],
                 weight_decay=1e-5,
                 charge_loss_weight=1.0,
                 interval_loss_weight=1.0
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

        self.output_projection_charge = nn.Linear(self.hparams.embed_dim, 1)
        self.output_projection_intervals = nn.Linear(self.hparams.embed_dim, 1)


    def encode(self, charges_log_norm_padded, times_log_norm_abs_padded, sensor_pos_padded, attention_mask):
        """Encodes input sequences into latent space parameters."""
        B, S = charges_log_norm_padded.shape
        device = charges_log_norm_padded.device

        # 1. Calculate summary statistics
        stats_dict = calculate_summary_stats(charges_log_norm_padded, times_log_norm_abs_padded, attention_mask)
        log_seq_length = stats_dict['log_seq_length']
        log_total_charge = stats_dict['log_total_charge']
        log_first_hit_time = stats_dict['log_first_hit_time']
        log_last_hit_time = stats_dict['log_last_hit_time']
        original_lengths = stats_dict['original_lengths']

        # 2. Convert absolute times to log-normalized intervals
        encoder_time_features_padded = convert_absolute_times_to_log_intervals(times_log_norm_abs_padded, original_lengths, attention_mask)

        # 3. Form summary stats tensor
        summary_stats_tensor = torch.stack([log_seq_length, log_total_charge, log_first_hit_time, log_last_hit_time], dim=-1)

        # 4. Embed input sequences
        concatenated_input = torch.stack((charges_log_norm_padded, encoder_time_features_padded), dim=-1).float()
        embedded_input = self.encoder_input_embedding(concatenated_input)

        # 5. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embedded_with_cls = torch.cat([cls_tokens, embedded_input], dim=1)
        embedded_with_cls = self.pos_encoder(embedded_with_cls)

        # 6. Prepare attention mask
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        extended_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # 7. Pass through transformer encoder
        encoded_sequence = self.transformer_encoder(embedded_with_cls, src_key_padding_mask=extended_attention_mask)
        
        # 8. Extract CLS token representation
        cls_representation = encoded_sequence[:, 0, :]
        
        # 9. Concatenate CLS representation with sensor position
        encoder_latent_input = torch.cat((cls_representation, sensor_pos_padded), dim=1)

        # 10. Project to content latent parameters
        mu_content = self.to_latent_mu(encoder_latent_input)
        logvar_content = self.to_latent_logvar(encoder_latent_input)

        # 11. Clamp logvar for stability
        logvar_content = torch.clamp(logvar_content, min=-10, max=10)

        # 12. Construct full mu and logvar
        mu_full = torch.cat([summary_stats_tensor, mu_content], dim=-1)
        deterministic_logvar = torch.full_like(summary_stats_tensor, -10.0)
        logvar_full = torch.cat([deterministic_logvar, logvar_content], dim=-1)

        return mu_full, logvar_full, stats_dict, encoder_time_features_padded


    def decode(self, z, inference=False):
        """Decodes latent vector z into reconstructed sequences."""
        B, device = z.size(0), z.device
        max_len = self.hparams.max_seq_len_padding

        # --- 1. Unpack Summary Stats from Latent Vector ---
        summary_stats = z[:, :self.summary_stats_dim]
        log_seq_length_z, log_total_charge_z, log_t_first_z, log_t_last_z = summary_stats.T

        seq_lengths_continuous = torch.exp(log_seq_length_z).clamp(min=1.0, max=float(max_len))
        actual_seq_lengths = torch.floor(seq_lengths_continuous).long() if inference else torch.round(seq_lengths_continuous).long()

        # --- 2. Prepare Input for Transformer Decoder ---
        z_projected = self.decoder_latent_to_input_proj(z)
        decoder_input = z_projected.unsqueeze(1) + self.query_embed.unsqueeze(0)
        decoder_input = self.decoder_pos_encoder(decoder_input)

        # --- 3. Transformer Decoding ---
        valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < actual_seq_lengths.unsqueeze(1)
        transformed_sequence = self.decoder_transformer(src=decoder_input, src_key_padding_mask=~valid_mask)

        # --- 4. Reconstruct Charges via Softmax Partitioning ---
        raw_charge_scores = self.output_projection_charge(transformed_sequence).squeeze(-1)
        masked_charge_scores = raw_charge_scores.masked_fill(~valid_mask, -float('inf'))
        charge_probs = torch.softmax(masked_charge_scores, dim=-1).to(masked_charge_scores.dtype)
        actual_total_charge = torch.exp(log_total_charge_z) - 1.0
        reconstructed_charges = charge_probs * actual_total_charge.unsqueeze(-1)

        # --- 5. Reconstruct Time Intervals via L-1 Mask Partitioning ---
        raw_interval_scores = self.output_projection_intervals(transformed_sequence).squeeze(-1)
        interval_mask = (torch.arange(max_len, device=device).unsqueeze(0) < (actual_seq_lengths - 1).unsqueeze(1)) & (actual_seq_lengths.unsqueeze(1) > 1)
        masked_interval_scores = raw_interval_scores.masked_fill(~interval_mask, -float('inf'))
        
        interval_probs = torch.zeros_like(masked_interval_scores)
        valid_rows = ~torch.all(masked_interval_scores == -float('inf'), dim=-1)
        if valid_rows.any():
            interval_probs[valid_rows] = torch.softmax(masked_interval_scores[valid_rows], dim=-1).to(interval_probs.dtype)

        actual_t_first = torch.exp(log_t_first_z)
        total_duration = (torch.exp(log_t_last_z) - actual_t_first).clamp(min=1e-9)
        reconstructed_intervals = interval_probs * total_duration.unsqueeze(-1)

        # --- 6. Format Output ---
        if not inference:
            return {'reconstructed_charges': reconstructed_charges, 'reconstructed_intervals': reconstructed_intervals}
        else:
            output_tq_pairs = []
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
        charges_log_norm_padded = batch['charges_log_norm_padded'].float()
        times_log_norm_abs_padded = batch['times_log_norm_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        sensor_pos_padded = batch['sensor_pos_batched'].float()

        mu, logvar, stats_dict, encoder_time_features_target = self.encode(
            charges_log_norm_padded, times_log_norm_abs_padded, sensor_pos_padded, attention_mask
        )

        z = reparameterize(mu, logvar)
        decoded_output = self.decode(z, inference=False)

        return {
            'mu': mu,
            'logvar': logvar,
            'reconstructed_charges': decoded_output['reconstructed_charges'],
            'reconstructed_intervals': decoded_output['reconstructed_intervals'],
            'original_charges_log_norm_padded': charges_log_norm_padded,
            'original_intervals_log_norm_padded': encoder_time_features_target,
            'original_lengths': stats_dict['original_lengths']
        }

    def _calculate_loss(self, forward_output):
        """Calculate reconstruction and KL divergence losses."""
        mu_full = forward_output['mu']
        logvar_full = forward_output['logvar']
        reconstructed_charges = forward_output['reconstructed_charges']
        reconstructed_intervals = forward_output['reconstructed_intervals']
        original_charges_log_norm_padded = forward_output['original_charges_log_norm_padded']
        original_intervals_log_norm_padded = forward_output['original_intervals_log_norm_padded']
        original_lengths = forward_output['original_lengths']

        device = mu_full.device
        B, max_len = reconstructed_charges.shape

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

        # Charge Loss
        pred_charges_log_norm = torch.log1p(reconstructed_charges)
        charge_losses = F.smooth_l1_loss(pred_charges_log_norm, original_charges_log_norm_padded, reduction='none')
        charge_recon_loss = (charge_losses * charge_valid_mask).sum() / (charge_valid_mask.sum() + 1e-8)

        # Interval Loss
        interval_valid_mask = (charge_valid_mask[:, :-1]) & (original_lengths.unsqueeze(1) > 1)
        if interval_valid_mask.any():
            pred_intervals_log_norm = torch.log(reconstructed_intervals[:, :-1].clamp(min=1e-9))
            target_intervals = original_intervals_log_norm_padded[:, :-1]
            interval_losses = F.smooth_l1_loss(pred_intervals_log_norm, target_intervals, reduction='none')
            interval_recon_loss = (interval_losses * interval_valid_mask).sum() / (interval_valid_mask.sum() + 1e-8)
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
        
        z = reparameterize(forward_output['mu'], forward_output['logvar'])
        mmd_loss = calculate_mmd_loss(z[:, self.summary_stats_dim:])
        
        total_loss = (self.hparams.charge_loss_weight * losses['charge_recon_loss'] +
                      self.hparams.interval_loss_weight * losses['interval_recon_loss'] +
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
        
        total_loss = (self.hparams.charge_loss_weight * losses['charge_recon_loss'] +
                      self.hparams.interval_loss_weight * losses['interval_recon_loss'] +
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
        
        total_loss = (self.hparams.charge_loss_weight * losses['charge_recon_loss'] +
                      self.hparams.interval_loss_weight * losses['interval_recon_loss'] +
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