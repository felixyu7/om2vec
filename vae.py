import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import PositionalEncoding
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

class NT_VAE(pl.LightningModule):
    def __init__(self,
                 latent_dim=128,
                 embed_dim=32, # Used for Transformer encoder d_model
                 beta_factor=1e-5,
                 beta_peak_epoch=4,
                 max_seq_len_padding=512,
                 transformer_encoder_layers=6,
                 transformer_encoder_heads=8,
                 transformer_encoder_ff_dim=2048, # d_model * 4 is common
                 transformer_encoder_dropout=0.1,
                 # Decoder HParams (matching config defaults)
                 transformer_decoder_layers=4,
                 transformer_decoder_heads=8,
                 transformer_decoder_ff_dim=256,
                 transformer_decoder_dropout=0.1,
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[2, 20],
                 weight_decay=1e-5
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: PyTorch Transformer based
        self.encoder_input_embedding = nn.Linear(2, self.hparams.embed_dim) # Input: (time, count)
        self.pos_encoder = PositionalEncoding(self.hparams.embed_dim,
                                              self.hparams.transformer_encoder_dropout,
                                              max_len=self.hparams.max_seq_len_padding + 1) # +1 for EOS token
        
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
        
        self.encoder_rnn = nn.GRU(
            input_size=self.hparams.embed_dim,
            hidden_size=self.hparams.embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.encoder_to_latent_input_dim = 2 * self.hparams.embed_dim + 3
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)
        
        # --- Non-autoregressive Decoder Components ---
        self.decoder_z_proj = nn.Linear(self.hparams.latent_dim, self.hparams.embed_dim)
        self.decoder_pos_encoder = PositionalEncoding(
            self.hparams.embed_dim,
            self.hparams.transformer_decoder_dropout,
            max_len=self.hparams.max_seq_len_padding
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
        self.decoder_transformer_encoder = nn.TransformerEncoder(
            decoder_encoder_layer,
            num_layers=self.hparams.transformer_decoder_layers
        )
        self.decoder_out_proj = nn.Linear(self.hparams.embed_dim, 3)

        self.beta = 0.
        self.current_train_iter = 0 # Renamed from self.iter for clarity

        # EOS loss weight hyperparameter
        self.hparams.eos_loss_weight = getattr(self.hparams, "eos_loss_weight", 1.0)

        self.test_step_results = {'num_hits': [], 'js_divs': []}

    def encode(self, times_data, counts_data, attention_mask, sensor_pos_batched):
        # times_data, counts_data: (B, S)
        # attention_mask: (B, S), boolean, True for padding tokens (False for valid)
        # sensor_pos_batched: (B, 3)
        
        concatenated_input = torch.stack((times_data, counts_data), dim=-1).float() # (B, S, 2)
        embedded_input = self.encoder_input_embedding(concatenated_input) # (B, S, embed_dim)
        embedded_input = self.pos_encoder(embedded_input) # Add positional encoding (B,S,E)
        
        # PyTorch TransformerEncoder expects src_key_padding_mask where True means PADDED/MASKED
        # Now attention_mask is True for padding tokens (False for valid), so pass directly.
        src_key_padding_mask = attention_mask # (B, S)
        
        encoded_sequence = self.transformer_encoder(embedded_input, src_key_padding_mask=src_key_padding_mask) # (B, S, embed_dim)
        
        # Pack the sequence for the RNN using the attention mask
        lengths = (~attention_mask).sum(dim=1).cpu()
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            encoded_sequence, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.encoder_rnn(packed_sequence)
        # hidden: (num_layers * num_directions, B, hidden_size)
        # For BiGRU, concatenate the last hidden states from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) # (B, 2*hidden_size)

        # Concatenate last_hidden with sensor_pos_batched
        encoder_latent_input = torch.cat((last_hidden, sensor_pos_batched), dim=1) # (B, 2*hidden_size + 3)

        mu = self.to_latent_mu(encoder_latent_input)
        logvar = self.to_latent_logvar(encoder_latent_input)

        # clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruction_loss(self, target_log_t, target_log_q, z, attention_mask, target_eos, sequence_lengths):
        """
        Calculates reconstruction loss for the non-autoregressive decoder.
        Args:
            target_log_t: (B, S_padded) log-normalized arrival times (ground truth)
            target_log_q: (B, S_padded) log-normalized charges (ground truth)
            z: (B, latent_dim)
            attention_mask: (B, S_padded) boolean, True for padding tokens (False for valid tokens in target)
            target_eos: (B, S_padded) float, EOS targets (0 or 1, 1 at actual EOS, 0 elsewhere within seq, 0 for padding)
            sequence_lengths: (B,) int, true sequence lengths (including EOS token position)
        Returns:
            mse_t: MSE for t
            mse_q: MSE for q
            eos_loss: BCE loss for EOS prediction
        """
        B, S_padded = target_log_t.shape
        device = target_log_t.device

        # Get full sequence predictions from the non-autoregressive decoder
        logits = self._decode_non_autoregressive(z)

        # Slice predictions to match batch's S_padded length for loss calculation
        pred_log_t_batch = logits[..., 0][:, :S_padded]  # (B, S_padded)
        pred_log_q_batch = logits[..., 1][:, :S_padded]  # (B, S_padded)
        pred_eos_logits_batch = logits[..., 2][:, :S_padded] # (B, S_padded)

        valid_token_mask = ~attention_mask # (B, S_padded), True for non-padded tokens

        # --- (log_t, log_q) Reconstruction Loss ---
        # Mask for actual data points (valid tokens AND not EOS position)
        data_points_mask = torch.zeros_like(valid_token_mask, dtype=torch.bool)
        for i in range(B):
            actual_data_len = sequence_lengths[i].item() - 1 # Data is up to index before EOS
            if actual_data_len > 0:
                data_points_mask[i, :actual_data_len] = True

        tq_loss_mask = valid_token_mask & data_points_mask

        if tq_loss_mask.any():
            mse_t = F.mse_loss(pred_log_t_batch[tq_loss_mask], target_log_t[tq_loss_mask])
            mse_q = F.mse_loss(pred_log_q_batch[tq_loss_mask], target_log_q[tq_loss_mask])
        else:
            mse_t = torch.tensor(0.0, device=device, requires_grad=True)
            mse_q = torch.tensor(0.0, device=device, requires_grad=True)

        # --- EOS Prediction Loss ---
        # Calculated over all valid (non-padded) token positions in the target sequence.
        if valid_token_mask.any():
            eos_loss = F.binary_cross_entropy_with_logits(
                pred_eos_logits_batch[valid_token_mask],
                target_eos[valid_token_mask], # target_eos is 0/1
                reduction='mean'
            )
        else:
            eos_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return mse_t, mse_q, eos_loss
    

    def _decode_non_autoregressive(self, z):
        """Internal helper for non-autoregressive decoding logic."""
        B = z.size(0)
        # 1. Project z and expand to sequence length
        z_dec_input = self.decoder_z_proj(z) # (B, embed_dim)
        decoder_input_sequence = z_dec_input.unsqueeze(1).repeat(1, self.hparams.max_seq_len_padding, 1)
        # 2. Add positional encoding
        decoder_input_sequence = self.decoder_pos_encoder(decoder_input_sequence)
        # 3. Pass through Transformer Encoder stack
        transformed_sequence = self.decoder_transformer_encoder(decoder_input_sequence) # (B, max_seq_len_padding, embed_dim)
        # 4. Project to output logits
        logits = self.decoder_out_proj(transformed_sequence) # (B, max_seq_len_padding, 3)
        return logits

    def decode(self, z):
        """
        Non-autoregressively generate a sequence of (log t, log q) given latent z.
        Args:
            z: (B, latent_dim)
        Returns:
            generated_sequences: (B, L_i, 2) log-normalised (t, q) sequences, where L_i is determined by EOS.
                                 Padded to the max length among L_i in the batch.
        """
        B, device = z.size(0), z.device

        logits = self._decode_non_autoregressive(z) # (B, max_seq_len_padding, 3)

        pred_log_t_full = logits[..., 0]  # (B, max_seq_len_padding)
        pred_log_q_full = logits[..., 1]  # (B, max_seq_len_padding)
        pred_eos_logits_full = logits[..., 2] # (B, max_seq_len_padding)

        eos_probs = torch.sigmoid(pred_eos_logits_full) # (B, max_seq_len_padding)

        # Determine sequence lengths (sampling EOS for generation)
        eos_samples = torch.bernoulli(eos_probs).bool() # (B, max_len)
        eos_samples_with_sentinel = torch.cat([eos_samples, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)
        predicted_lengths = torch.argmax(eos_samples_with_sentinel.int(), dim=1) + 1
        predicted_lengths = torch.clamp(predicted_lengths, min=1, max=self.hparams.max_seq_len_padding) # Ensure min length 1

        output_sequences = []
        max_len_batch = 0
        for i in range(B):
            length = predicted_lengths[i].item() # This length includes the EOS token position
            # Sequence of (log_t, log_q) up to and including the EOS position
            current_seq = torch.stack([pred_log_t_full[i, :length],
                                       pred_log_q_full[i, :length]], dim=-1) # (length, 2)
            output_sequences.append(current_seq)
            if length > max_len_batch:
                max_len_batch = length

        generated_padded = torch.zeros(B, max_len_batch, 2, device=device)
        for i, seq in enumerate(output_sequences):
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

    def kl_divergence(self, mu, logvar, free_bits_lambda=0.1):
        # Per-dimension KL: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        # Apply free-bits threshold
        kl_per_dim = F.relu(kl_per_dim - free_bits_lambda)
        # Sum over latent dims, mean over batch
        return kl_per_dim.sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        target_eos = batch['eos_target_padded'].float()
        sequence_lengths = batch['sequence_lengths']

        # Beta annealing logic remains
        if not hasattr(self, "steps_initialized") or not self.steps_initialized:
            if self.trainer and hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader:
                steps_per_epoch = len(self.trainer.train_dataloader)
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch
            else:
                self.total_steps_for_beta_annealing = 1
            self.steps_initialized = True

        mse_t, mse_q, eos_loss = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths
        )
        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge + self.hparams.eos_loss_weight * eos_loss
        kl_loss = self.kl_divergence(mu, logvar)
        if self.total_steps_for_beta_annealing > 0:
            progress_ratio = min(self.current_train_iter / self.total_steps_for_beta_annealing, 1.0)
            self.beta = self.hparams.beta_factor * ((np.cos(np.pi * (progress_ratio - 1)) + 1) / 2)
        else:
            self.beta = self.hparams.beta_factor
        loss = reconstruction_loss + (self.beta * kl_loss)
        self.current_train_iter += 1
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("train_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_eos_loss", eos_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        target_eos = batch['eos_target_padded'].float()
        sequence_lengths = batch['sequence_lengths']

        mse_t, mse_q, eos_loss = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths
        )

        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge + self.hparams.eos_loss_weight * eos_loss

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)

        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_eos_loss", eos_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        target_eos = batch['eos_target_padded'].float()
        sequence_lengths = batch['sequence_lengths']

        mse_t, mse_q, eos_loss = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths
        )

        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge + self.hparams.eos_loss_weight * eos_loss

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)

        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_eos_loss", eos_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[0], eta_min=self.hparams.lr_schedule[1])
        return [optimizer], [scheduler]