import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from utils import PositionalEncoding

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
                 interval_loss_weight=1.0,
                 first_time_loss_weight=1.0,
                 length_loss_weight=1.0
                 ):
        super().__init__()
        self.save_hyperparameters()

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

        self.encoder_to_latent_input_dim = self.hparams.embed_dim + 3
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)

        # --- Transformer Encoder based Decoder ---
        self.decoder_latent_to_input_proj = nn.Linear(self.hparams.latent_dim, self.hparams.embed_dim)
        self.decoder_pos_encoder = PositionalEncoding(
            self.hparams.embed_dim,
            self.hparams.transformer_decoder_dropout, # Configures dropout for the decoder's TransformerEncoder
            max_len=self.hparams.max_seq_len_padding
        )
        # Learnable content embeddings for each decoder position
        self.query_embed = nn.Parameter(
            torch.randn(self.hparams.max_seq_len_padding, self.hparams.embed_dim) * 0.02
        )
        # Input to the decoder's transformer is derived from z + positional content embeds + pos encoding.

        decoder_encoder_layer = nn.TransformerEncoderLayer( # Changed from TransformerDecoderLayer
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.transformer_decoder_heads,    # Configures heads for the decoder's TransformerEncoder
            dim_feedforward=self.hparams.transformer_decoder_ff_dim, # Configures ff_dim for the decoder's TransformerEncoder
            activation='gelu',
            dropout=self.hparams.transformer_decoder_dropout, # Reused: dropout for the decoder's TransformerEncoder
            batch_first=True,
            norm_first=True
        )
        self.decoder_transformer = nn.TransformerEncoder( # Changed from TransformerDecoder
            decoder_encoder_layer,
            num_layers=self.hparams.transformer_decoder_layers # Configures layers for the decoder's TransformerEncoder
        )

        # Decoder outputs: charges and relative time intervals
        self.decoder_out_proj_charges = nn.Linear(self.hparams.embed_dim, 1)
        self.decoder_out_proj_intervals = nn.Linear(self.hparams.embed_dim, 1)

        # Length predictor head: outputs log-normalized sequence length
        self.length_predictor = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.latent_dim, 1)
        )
        
        # First absolute time predictor head: outputs log-normalized first absolute time
        self.first_abs_time_predictor = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.latent_dim, 1)
        )

        self.beta = 0.
        self.current_train_iter = 0
        self.test_step_results = {'num_hits': [], 'js_divs': []}

    def encode(self, q_sequence_padded, time_input_features_padded, src_key_padding_mask, sensor_pos_batched):
        # q_sequence_padded: (B, S) - log-normalized charges
        # time_input_features_padded: (B, S) - log-normalized time features for encoder input
        # src_key_padding_mask: (B, S), boolean, True for padding tokens (False for valid)
        # sensor_pos_batched: (B, 3)
        B, S = q_sequence_padded.shape
        device = q_sequence_padded.device

        # 1. Embed input sequences (combine charges and time features)
        concatenated_input = torch.stack((q_sequence_padded, time_input_features_padded), dim=-1).float() # (B, S, 2)
        embedded_input = self.encoder_input_embedding(concatenated_input) # (B, S, embed_dim)

        # 2. Prepend CLS token to each sequence
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        embedded_with_cls = torch.cat([cls_tokens, embedded_input], dim=1) # (B, 1+S, embed_dim)

        # 3. Apply positional encoding
        embedded_with_cls = self.pos_encoder(embedded_with_cls) # (B, 1+S, embed_dim)

        # 4. Prepare attention mask for CLS token + sequence
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device) # (B, 1)
        extended_attention_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1) # (B, 1+S)

        # 5. Pass through transformer encoder
        encoded_sequence = self.transformer_encoder(
            embedded_with_cls,
            src_key_padding_mask=extended_attention_mask
        ) # (B, 1+S, embed_dim)

        # 6. Extract CLS token representation (first token)
        cls_representation = encoded_sequence[:, 0, :] # (B, embed_dim)

        # 7. Concatenate CLS representation with sensor position
        encoder_latent_input = torch.cat((cls_representation, sensor_pos_batched), dim=1) # (B, embed_dim + 3)

        # 8. Project to latent parameters (no length reserved)
        mu = self.to_latent_mu(encoder_latent_input) # (B, latent_dim)
        logvar = self.to_latent_logvar(encoder_latent_input) # (B, latent_dim)

        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_reconstruction_losses(self, predictions, batch):
        """
        Compute reconstruction losses for charges, intervals, first absolute time, and length.
        """
        device = predictions['z'].device
        
        # Extract predictions
        pred_q_sequence = predictions['pred_q_sequence']  # (B, max_seq_len)
        pred_log_intervals = predictions['pred_log_intervals']  # (B, max_seq_len)
        pred_log_first_abs_time = predictions['pred_log_first_abs_time']  # (B,)
        pred_log_length = predictions['pred_log_length']  # (B,)
        
        # Extract targets
        target_q_sequence = batch['q_sequence_padded'].float()  # (B, max_seq_len)
        target_log_intervals = batch['target_log_intervals_padded'].float()  # (B, max_interval_len)
        target_log_first_abs_time = batch['target_log_first_abs_time'].float()  # (B,)
        sequence_lengths = batch['sequence_lengths']  # (B,)
        
        # Extract masks
        # For loss calculation, masks should be True for VALID elements.
        # Batch's src_key_padding_mask/attention_mask is False for VALID, True for PADDING (from data_utils.py).
        # So, we invert it for loss calculation.
        _src_mask_from_batch = batch.get('src_key_padding_mask', batch['attention_mask']).bool() # False=valid, True=pad
        src_mask_for_loss = ~_src_mask_from_batch  # Now True=valid, False=pad
        # Batch's interval_mask is False for VALID, True for PADDING (from data_utils.py).
        # So, we invert it for loss calculation.
        _interval_mask_from_batch = batch['interval_mask'].bool() # False=valid, True=pad
        interval_mask_for_loss = ~_interval_mask_from_batch # Now True=valid, False=pad
        
        # 1. Charge reconstruction loss
        B, S_padded = target_q_sequence.shape
        len_for_loss = min(pred_q_sequence.shape[1], S_padded)
        
        pred_q_sliced = pred_q_sequence[:, :len_for_loss]
        target_q_sliced = target_q_sequence[:, :len_for_loss]
        src_mask_for_loss_sliced = src_mask_for_loss[:, :len_for_loss]
        
        # Create valid mask for charges
        charge_valid_mask = torch.zeros_like(target_q_sliced, dtype=torch.bool)
        # Create range tensor and compare with sequence lengths
        range_tensor = torch.arange(len_for_loss, device=device).unsqueeze(0).expand(B, -1)  # (B, len_for_loss)
        sequence_lengths_expanded = sequence_lengths.unsqueeze(1).expand(-1, len_for_loss)  # (B, len_for_loss)
        charge_valid_mask = (range_tensor < sequence_lengths_expanded) & (sequence_lengths_expanded > 0)
        # charge_valid_mask is True for valid elements based on sequence_lengths.
        # src_mask_for_loss_sliced is also True for valid elements from input mask.
        charge_valid_mask = charge_valid_mask & src_mask_for_loss_sliced
        
        if charge_valid_mask.any():
            charge_loss = F.mse_loss(pred_q_sliced[charge_valid_mask], target_q_sliced[charge_valid_mask])
        else:
            charge_loss = torch.tensor(0.0, device=device) # No requires_grad=True for non-graph-connected tensor
        
        # 2. Interval reconstruction loss
        interval_len_for_loss = min(pred_log_intervals.shape[1], target_log_intervals.shape[1])
        pred_intervals_sliced = pred_log_intervals[:, :interval_len_for_loss]
        target_intervals_sliced = target_log_intervals[:, :interval_len_for_loss]
        interval_mask_for_loss_sliced = interval_mask_for_loss[:, :interval_len_for_loss]
        
        # Valid intervals mask (intervals exist for sequences of length > 1)
        # interval_mask_for_loss_sliced is True for valid intervals.
        interval_valid_mask = interval_mask_for_loss_sliced
        
        if interval_valid_mask.any():
            interval_loss = F.mse_loss(pred_intervals_sliced[interval_valid_mask], target_intervals_sliced[interval_valid_mask])
        else:
            interval_loss = torch.tensor(0.0, device=device) # No requires_grad=True for non-graph-connected tensor
        
        # 3. First absolute time loss
        first_time_loss = F.mse_loss(pred_log_first_abs_time, target_log_first_abs_time)
        
        # 4. Length prediction loss
        target_log_lengths = torch.log(sequence_lengths.clamp(min=1, max=self.hparams.max_seq_len_padding).float())
        length_loss = F.mse_loss(pred_log_length, target_log_lengths)
        
        return {
            'charge_loss': charge_loss,
            'interval_loss': interval_loss,
            'first_time_loss': first_time_loss,
            'length_loss': length_loss
        }

    def _decode_non_autoregressive(self, z):
        B = z.size(0)
        device = z.device

        # 1. Project latent z and expand to sequence length for Transformer Encoder input
        z_projected = self.decoder_latent_to_input_proj(z)  # (B, embed_dim)
        
        # Expand z_projected and add learnable positional content embeddings
        # z_projected_expanded: (B, max_seq_len_padding, embed_dim)
        z_projected_expanded = z_projected.unsqueeze(1).expand(-1, self.hparams.max_seq_len_padding, -1)
        
        # query_embed: (max_seq_len_padding, embed_dim)
        # Add to z_projected_expanded (broadcasts across batch)
        decoder_input_sequence = z_projected_expanded + self.query_embed.unsqueeze(0) # (B, max_seq_len_padding, embed_dim)
        
        # 2. Add sinusoidal positional encoding to the combined decoder input sequence
        decoder_input_sequence = self.decoder_pos_encoder(decoder_input_sequence)  # (B, max_seq_len_padding, embed_dim)

        # 3. Pass through Transformer Encoder (acting as the decoder)
        # This uses self-attention on the sequence derived from z + learnable content + pos_enc.
        transformed_sequence = self.decoder_transformer(
            src=decoder_input_sequence
        )  # (B, max_seq_len_padding, embed_dim)

        # 5. Project to outputs: charges and intervals separately
        pred_charges = self.decoder_out_proj_charges(transformed_sequence)  # (B, max_seq_len_padding, 1)
        pred_intervals = self.decoder_out_proj_intervals(transformed_sequence)  # (B, max_seq_len_padding, 1)
        
        pred_charges = pred_charges.squeeze(-1)  # (B, max_seq_len_padding)
        pred_intervals = pred_intervals.squeeze(-1)  # (B, max_seq_len_padding)
        
        return pred_charges, pred_intervals

    def decode(self, z, target_len=None, first_abs_time_override=None):
        """
        Non-autoregressively generate reconstructed absolute time and charge sequences.
        Args:
            z: (B, latent_dim)
            target_len: (B,) or None. If provided, use as output length. If None, predict length from z.
            first_abs_time_override: (B,) or None. If provided, use instead of predicting from z.
        Returns:
            reconstructed_sequences: (B, L_i, 2) absolute (t, q) sequences, where L_i is determined by target_len.
        """
        B, device = z.size(0), z.device

        # Predict or use provided sequence lengths
        if target_len is not None:
            predicted_lengths = target_len.long().clamp(min=1, max=self.hparams.max_seq_len_padding)
        else:
            log_length = self.length_predictor(z).squeeze(-1)  # (B,)
            predicted_lengths = torch.exp(log_length).round().long()
            predicted_lengths = predicted_lengths.clamp(min=1, max=self.hparams.max_seq_len_padding)

        # Predict first absolute time or use override
        if first_abs_time_override is not None:
            pred_log_first_abs_time = torch.log(first_abs_time_override + 1e-8)  # (B,)
        else:
            pred_log_first_abs_time = self.first_abs_time_predictor(z).squeeze(-1)  # (B,)

        # Get sequence predictions (charges and intervals)
        pred_charges_full, pred_intervals_full = self._decode_non_autoregressive(z)

        # Reconstruct absolute time sequences
        output_sequences = []
        max_len_batch = 0
        for i in range(B):
            length = predicted_lengths[i].item()
            if length == 0:
                continue
                
            # Get predicted charges for this sequence
            pred_q_seq = pred_charges_full[i, :length]  # (length,)
            
            # Reconstruct absolute times from first time + intervals
            recon_first_abs_t = torch.exp(pred_log_first_abs_time[i])  # scalar
            recon_t_abs = torch.zeros(length, device=device)
            recon_t_abs[0] = recon_first_abs_t
            
            if length > 1:
                # Get intervals for reconstruction (length-1)
                pred_intervals_seq = pred_intervals_full[i, :length-1]  # (length-1,)
                recon_intervals = torch.exp(pred_intervals_seq)  # (length-1,)
                
                # Cumulative sum to get absolute times
                recon_t_abs[1:] = recon_first_abs_t + torch.cumsum(recon_intervals, dim=0)
            
            # Denormalize charges
            recon_q_seq = torch.exp(pred_q_seq) - 1  # Inverse of log1p
            
            current_seq = torch.stack([recon_t_abs, recon_q_seq], dim=-1)  # (length, 2)
            output_sequences.append(current_seq)
            if length > max_len_batch:
                max_len_batch = length

        if max_len_batch == 0:
            max_len_batch = 1

        generated_padded = torch.zeros(B, max_len_batch, 2, device=device)
        for i, seq in enumerate(output_sequences):
            if seq.shape[0] > 0:
                generated_padded[i, :seq.shape[0], :] = seq

        return generated_padded

    def forward(self, batch):
        q_sequence_padded = batch['q_sequence_padded'].float()
        time_input_features_padded = batch['time_input_features_padded'].float()
        # Batch's src_key_padding_mask/attention_mask is False for VALID, True for PADDING (from data_utils.py).
        # PyTorch Transformer's src_key_padding_mask also expects True for PADDING tokens.
        # So, the mask from batch can be used directly.
        transformer_src_key_padding_mask = batch.get('src_key_padding_mask', batch['attention_mask']).bool()
        sensor_pos_batched = batch['sensor_pos_batched'].float() # (B, 3)

        mu, logvar = self.encode(q_sequence_padded, time_input_features_padded, transformer_src_key_padding_mask, sensor_pos_batched)
        z = self.reparameterize(mu, logvar) # (B, latent_dim)

        # Predict length and first absolute time
        pred_log_length = self.length_predictor(z).squeeze(-1)  # (B,)
        pred_log_first_abs_time = self.first_abs_time_predictor(z).squeeze(-1)  # (B,)

        # Get sequence predictions
        pred_q_sequence, pred_log_intervals = self._decode_non_autoregressive(z)

        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'pred_log_length': pred_log_length,
            'pred_log_first_abs_time': pred_log_first_abs_time,
            'pred_q_sequence': pred_q_sequence,
            'pred_log_intervals': pred_log_intervals
        }

    def kl_divergence(self, mu, logvar):
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        return kl_per_dim.sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        mu, logvar = predictions['mu'], predictions['logvar']

        if not hasattr(self, "steps_initialized") or not self.steps_initialized:
            steps_per_epoch = 100  # Default steps per epoch
            if self.trainer and hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader:
                if hasattr(self.trainer.train_dataloader, "__len__"):
                    try:
                        current_dataloader_len = len(self.trainer.train_dataloader)
                        if current_dataloader_len > 0:
                            steps_per_epoch = current_dataloader_len
                        # If len is 0 or less, default of 100 is used.
                    except TypeError:
                        # len() failed or not supported, default of 100 is used.
                        pass
                # If no __len__ attribute, default of 100 is used.
            self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch
            self.steps_initialized = True

        # Compute reconstruction losses
        losses = self.compute_reconstruction_losses(predictions, batch)
        
        # KL divergence loss
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Beta annealing
        if self.total_steps_for_beta_annealing > 0:
            progress_ratio = min(self.current_train_iter / self.total_steps_for_beta_annealing, 1.0)
            self.beta = self.hparams.beta_factor * ((np.cos(np.pi * (progress_ratio - 1)) + 1) / 2)
        else:
            self.beta = self.hparams.beta_factor

        # Total reconstruction loss with weights
        reconstruction_loss = (self.hparams.charge_loss_weight * losses['charge_loss'] +
                             self.hparams.interval_loss_weight * losses['interval_loss'] +
                             self.hparams.first_time_loss_weight * losses['first_time_loss'] +
                             self.hparams.length_loss_weight * losses['length_loss'])
        
        # Total loss
        loss = reconstruction_loss + (self.beta * kl_loss)
        
        self.current_train_iter += 1
        
        # Logging
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("train_reco_loss_charge", losses['charge_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_interval", losses['interval_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_first_time", losses['first_time_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("length_loss", losses['length_loss'], batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        mu, logvar = predictions['mu'], predictions['logvar']

        # Compute reconstruction losses
        losses = self.compute_reconstruction_losses(predictions, batch)
        
        # KL divergence loss
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Total reconstruction loss with weights
        reconstruction_loss = (self.hparams.charge_loss_weight * losses['charge_loss'] +
                             self.hparams.interval_loss_weight * losses['interval_loss'] +
                             self.hparams.first_time_loss_weight * losses['first_time_loss'] +
                             self.hparams.length_loss_weight * losses['length_loss'])
        
        # Total loss
        loss = reconstruction_loss + (self.beta * kl_loss)
        
        # Logging
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_reco_loss_charge", losses['charge_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_interval", losses['interval_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_first_time", losses['first_time_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_length_loss", losses['length_loss'], batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        predictions = self(batch)
        mu, logvar = predictions['mu'], predictions['logvar']

        # Compute reconstruction losses
        losses = self.compute_reconstruction_losses(predictions, batch)
        
        # KL divergence loss
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Total reconstruction loss with weights
        reconstruction_loss = (self.hparams.charge_loss_weight * losses['charge_loss'] +
                             self.hparams.interval_loss_weight * losses['interval_loss'] +
                             self.hparams.first_time_loss_weight * losses['first_time_loss'] +
                             self.hparams.length_loss_weight * losses['length_loss'])
        
        # Total loss
        loss = reconstruction_loss + (self.beta * kl_loss)
        
        # Logging
        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("test_reco_loss_charge", losses['charge_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_interval", losses['interval_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_first_time", losses['first_time_loss'], batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_length_loss", losses['length_loss'], batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.lr_schedule[0],
            eta_min=self.hparams.lr_schedule[1]
        )
        return [optimizer], [scheduler]