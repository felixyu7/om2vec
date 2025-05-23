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
                 weight_decay=1e-5
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: PyTorch Transformer based with CLS token
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
        # Output dim is now latent_dim (no reserved slot for length)
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)

        # --- Cross-Attention Decoder with Latent Token ---
        # Project latent z to a single memory token for cross-attention
        self.decoder_z_to_memory = nn.Linear(self.hparams.latent_dim, self.hparams.embed_dim)
        self.decoder_pos_encoder = PositionalEncoding(
            self.hparams.embed_dim,
            self.hparams.transformer_decoder_dropout,
            max_len=self.hparams.max_seq_len_padding
        )
        # Learnable query embeddings for the decoder input
        self.decoder_query_embeds = nn.Parameter(
            torch.randn(self.hparams.max_seq_len_padding, self.hparams.embed_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.transformer_decoder_heads,
            dim_feedforward=self.hparams.transformer_decoder_ff_dim,
            activation='gelu',
            dropout=self.hparams.transformer_decoder_dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder_transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.hparams.transformer_decoder_layers
        )

        self.decoder_out_proj = nn.Linear(self.hparams.embed_dim, 2)

        # Length predictor head: outputs log-normalized sequence length
        self.length_predictor = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hparams.latent_dim // 2, 1)
        )

        self.beta = 0.
        self.current_train_iter = 0
        self.test_step_results = {'num_hits': [], 'js_divs': []}

    def encode(self, times_data, counts_data, attention_mask, sensor_pos_batched):
        # times_data, counts_data: (B, S)
        # attention_mask: (B, S), boolean, True for padding tokens (False for valid)
        # sensor_pos_batched: (B, 3)
        B, S = times_data.shape
        device = times_data.device

        # 1. Embed input sequences
        concatenated_input = torch.stack((times_data, counts_data), dim=-1).float() # (B, S, 2)
        embedded_input = self.encoder_input_embedding(concatenated_input) # (B, S, embed_dim)

        # 2. Prepend CLS token to each sequence
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        embedded_with_cls = torch.cat([cls_tokens, embedded_input], dim=1) # (B, 1+S, embed_dim)

        # 3. Apply positional encoding
        embedded_with_cls = self.pos_encoder(embedded_with_cls) # (B, 1+S, embed_dim)

        # 4. Prepare attention mask for CLS token + sequence
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device) # (B, 1)
        extended_attention_mask = torch.cat([cls_mask, attention_mask], dim=1) # (B, 1+S)

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

    def reconstruction_loss(self, target_log_t, target_log_q, z, attention_mask, true_sequence_lengths):
        B, S_padded = target_log_t.shape
        device = target_log_t.device

        # Get sequence predictions using the decode method, respecting true_sequence_lengths
        # self.decode returns (B, L_max_in_batch, 2) where L_max_in_batch is determined by true_sequence_lengths
        predicted_log_tq_decoded = self.decode(z, true_seq_len=true_sequence_lengths) # (B, L_batch_max, 2)

        # Ensure predicted_log_tq_decoded is sliced or padded to match S_padded for consistent loss calculation
        len_for_loss = min(predicted_log_tq_decoded.shape[1], S_padded)

        pred_log_t_batch = predicted_log_tq_decoded[:, :len_for_loss, 0]  # (B, len_for_loss)
        pred_log_q_batch = predicted_log_tq_decoded[:, :len_for_loss, 1]  # (B, len_for_loss)
        
        # Target needs to be sliced to len_for_loss as well for comparison
        target_log_t_sliced = target_log_t[:, :len_for_loss]
        target_log_q_sliced = target_log_q[:, :len_for_loss]

        # Mask for actual data points based on true_sequence_lengths, applied to the sliced tensors
        data_points_mask = torch.zeros_like(target_log_t_sliced, dtype=torch.bool) # (B, len_for_loss)
        for i in range(B):
            actual_len = true_sequence_lengths[i].item()
            if actual_len > 0:
                data_points_mask[i, :actual_len] = True

        # attention_mask is (B, S_padded), slice it too
        attention_mask_sliced = attention_mask[:, :len_for_loss]
        valid_loss_mask = data_points_mask & (~attention_mask_sliced)

        if valid_loss_mask.any():
            mse_t = F.smooth_l1_loss(pred_log_t_batch[valid_loss_mask], target_log_t_sliced[valid_loss_mask])
            mse_q = F.smooth_l1_loss(pred_log_q_batch[valid_loss_mask], target_log_q_sliced[valid_loss_mask])
        else:
            mse_t = torch.tensor(0.0, device=device, requires_grad=True)
            mse_q = torch.tensor(0.0, device=device, requires_grad=True)

        return mse_t, mse_q

    def _decode_non_autoregressive(self, z):
        B = z.size(0)
        device = z.device

        # 1. Project z to a single memory token for cross-attention
        memory_token = self.decoder_z_to_memory(z)  # (B, embed_dim)
        memory_token = memory_token.unsqueeze(1)  # (B, 1, embed_dim)

        # 2. Create decoder input sequence using learnable query embeddings
        decoder_input_sequence = self.decoder_query_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, max_seq_len_padding, embed_dim)
        
        # 3. Add sinusoidal positional encoding to decoder input
        decoder_input_sequence = self.decoder_pos_encoder(decoder_input_sequence)  # (B, max_seq_len_padding, embed_dim)

        # 4. Pass through cross-attention decoder
        # Memory is just the single projected latent token
        transformed_sequence = self.decoder_transformer(
            tgt=decoder_input_sequence,
            memory=memory_token
        )  # (B, max_seq_len_padding, embed_dim)

        # 5. Project to output (log_t, log_q)
        predicted_log_tq = self.decoder_out_proj(transformed_sequence)  # (B, max_seq_len_padding, 2)
        return predicted_log_tq

    def decode(self, z, true_seq_len=None):
        """
        Non-autoregressively generate a sequence of (log t, log q) given latent z.
        Args:
            z: (B, latent_dim)
            true_seq_len: (B,) or None. If provided, use as output length. If None, predict length from z.
        Returns:
            generated_sequences: (B, L_i, 2) log-normalised (t, q) sequences, where L_i is determined by length predictor or true_seq_len.
                                 Padded to the max length among L_i in the batch.
        """
        B, device = z.size(0), z.device

        if true_seq_len is not None:
            # Use provided true sequence lengths (training)
            predicted_lengths = true_seq_len.long().clamp(min=1, max=self.hparams.max_seq_len_padding)
        else:
            # Predict length from z using the length predictor head (inference)
            log_length = self.length_predictor(z).squeeze(-1)  # (B,)
            predicted_lengths = torch.exp(log_length).round().long()  # Convert back from log space
            predicted_lengths = predicted_lengths.clamp(min=1, max=self.hparams.max_seq_len_padding)

        # Get full sequence predictions (log_t, log_q)
        predicted_log_tq_full = self._decode_non_autoregressive(z)

        pred_log_t_full = predicted_log_tq_full[..., 0]
        pred_log_q_full = predicted_log_tq_full[..., 1]

        output_sequences = []
        max_len_batch = 0
        for i in range(B):
            length = predicted_lengths[i].item()
            current_seq_log_t = pred_log_t_full[i, :length]
            current_seq_log_q = pred_log_q_full[i, :length]
            current_seq = torch.stack([current_seq_log_t, current_seq_log_q], dim=-1)
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
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        sensor_pos_batched = batch['sensor_pos_batched'].float() # (B, 3)

        mu, logvar = self.encode(times_padded, counts_padded, attention_mask, sensor_pos_batched)
        z = self.reparameterize(mu, logvar) # (B, latent_dim)

        return mu, logvar, z

    def kl_divergence(self, mu, logvar):
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        return kl_per_dim.sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        true_sequence_lengths = batch['sequence_lengths'].clamp(min=0)

        if not hasattr(self, "steps_initialized") or not self.steps_initialized:
            if self.trainer and hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader:
                steps_per_epoch = len(self.trainer.train_dataloader)
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch
            else:
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * 100
            self.steps_initialized = True

        mse_t, mse_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, true_sequence_lengths
        )
        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge
        kl_loss = self.kl_divergence(mu, logvar)
        if self.total_steps_for_beta_annealing > 0:
            progress_ratio = min(self.current_train_iter / self.total_steps_for_beta_annealing, 1.0)
            self.beta = self.hparams.beta_factor * ((np.cos(np.pi * (progress_ratio - 1)) + 1) / 2)
        else:
            self.beta = self.hparams.beta_factor

        # Length prediction loss (regression with MSE)
        # Target: log-normalized sequence lengths
        predicted_log_lengths = self.length_predictor(z).squeeze(-1)  # (B,)
        target_log_lengths = torch.log(true_sequence_lengths.clamp(min=1, max=self.hparams.max_seq_len_padding).float())  # (B,)
        length_loss = F.smooth_l1_loss(predicted_log_lengths, target_log_lengths)

        loss = reconstruction_loss + (self.beta * kl_loss) + length_loss
        self.current_train_iter += 1
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("train_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("length_loss", length_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        true_sequence_lengths = batch['sequence_lengths'].clamp(min=0)

        mse_t, mse_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, true_sequence_lengths
        )
        reconstruction_loss = mse_t + mse_q
        kl_loss = self.kl_divergence(mu, logvar)
        predicted_log_lengths = self.length_predictor(z).squeeze(-1)  # (B,)
        target_log_lengths = torch.log(true_sequence_lengths.clamp(min=1, max=self.hparams.max_seq_len_padding).float())  # (B,)
        length_loss = F.smooth_l1_loss(predicted_log_lengths, target_log_lengths)

        loss = reconstruction_loss + (self.beta * kl_loss) + length_loss
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_length_loss", length_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        true_sequence_lengths = batch['sequence_lengths'].clamp(min=0)

        mse_t, mse_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, true_sequence_lengths
        )
        reconstruction_loss = mse_t + mse_q
        kl_loss = self.kl_divergence(mu, logvar)
        predicted_log_lengths = self.length_predictor(z).squeeze(-1)  # (B,)
        target_log_lengths = torch.log(true_sequence_lengths.clamp(min=1, max=self.hparams.max_seq_len_padding).float())  # (B,)
        length_loss = F.smooth_l1_loss(predicted_log_lengths, target_log_lengths)

        loss = reconstruction_loss + (self.beta * kl_loss) + length_loss
        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("test_length_loss", length_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.lr_schedule[0],
            eta_min=self.hparams.lr_schedule[1]
        )
        return [optimizer], [scheduler]