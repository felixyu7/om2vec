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
                 memory_bank_size=256, # Size of learnable memory bank
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[2, 20],
                 weight_decay=1e-5
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: PyTorch Transformer based with CLS token
        self.encoder_input_embedding = nn.Linear(2, self.hparams.embed_dim) # Input: (time, count)
        
        # Learnable CLS token for sequence aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hparams.embed_dim) * 0.02)
        
        # Positional encoding with +1 for CLS token
        self.pos_encoder = PositionalEncoding(self.hparams.embed_dim,
                                              self.hparams.transformer_encoder_dropout,
                                              max_len=self.hparams.max_seq_len_padding + 1) # +1 for CLS token
        
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
        
        # CLS token approach replaces GRU - input dimension is now embed_dim + 3 (sensor pos)
        self.encoder_to_latent_input_dim = self.hparams.embed_dim + 3
        # Output dim is latent_dim - 1 because z[0] will be reserved for sequence length
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim - 1)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim - 1)
        
        # --- Cross-Attention Decoder with Memory Bank ---
        # Learnable memory bank that captures common patterns
        self.memory_bank = nn.Parameter(
            torch.randn(self.hparams.memory_bank_size, self.hparams.embed_dim) * 0.02
        )
        
        # Input dim is latent_dim - 1 as z[0] (length) is handled separately
        self.decoder_z_proj = nn.Linear(self.hparams.latent_dim - 1, self.hparams.embed_dim)
        self.decoder_pos_encoder = PositionalEncoding(
            self.hparams.embed_dim,
            self.hparams.transformer_decoder_dropout,
            max_len=self.hparams.max_seq_len_padding
        )
        
        # Cross-attention decoder layers
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
        
        # Output projection to (log_t, log_q)
        self.decoder_out_proj = nn.Linear(self.hparams.embed_dim, 2)

        self.beta = 0.
        self.current_train_iter = 0 # Renamed from self.iter for clarity

        # EOS related hparams removed

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
        # CLS token should never be masked (always attends to sequence)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device) # (B, 1) - False for CLS
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

        # 8. Project to latent parameters
        mu_content = self.to_latent_mu(encoder_latent_input) # (B, latent_dim - 1)
        logvar_content = self.to_latent_logvar(encoder_latent_input) # (B, latent_dim - 1)

        # Clamp logvar_content for stability
        logvar_content = torch.clamp(logvar_content, min=-10, max=10)

        # 9. Compute sequence lengths and log-lengths
        # Use original attention_mask to get actual sequence lengths
        lengths = (~attention_mask).sum(dim=1).float().clamp(min=1.0) # (B,)
        log_true_lengths = torch.log(lengths.to(mu_content.device)).unsqueeze(1) # (B, 1)
        
        # Log-variance for length dimension (fixed, small variance)
        log_true_lengths_logvar = torch.full_like(log_true_lengths, -10.0) # (B, 1)

        # 10. Combine length and content parameters
        mu = torch.cat((log_true_lengths, mu_content), dim=1) # (B, latent_dim)
        logvar = torch.cat((log_true_lengths_logvar, logvar_content), dim=1) # (B, latent_dim)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # mu and logvar are (B, latent_dim)
        # mu[:, 0] is log_true_length (not sampled)
        # mu[:, 1:] is content_mu
        # logvar[:, 0] is log_true_length_logvar (fixed, small)
        # logvar[:, 1:] is content_logvar
        
        log_lengths_mu = mu[:, 0:1] # (B, 1)
        content_mu = mu[:, 1:]      # (B, latent_dim - 1)
        
        # We don't sample length, but if we did, it would use logvar[:, 0:1]
        # For content part:
        content_logvar = logvar[:, 1:] # (B, latent_dim - 1)
        std_content = torch.exp(0.5 * content_logvar)
        eps_content = torch.randn_like(std_content)
        z_content = content_mu + eps_content * std_content # (B, latent_dim - 1)
        
        # Concatenate the deterministic log_length with the sampled content
        z = torch.cat((log_lengths_mu, z_content), dim=1) # (B, latent_dim)
        return z

    def reconstruction_loss(self, target_log_t, target_log_q, z, attention_mask, true_sequence_lengths):
        """
        Calculates reconstruction loss for the non-autoregressive decoder.
        Args:
            target_log_t: (B, S_padded) log-normalized arrival times (ground truth)
            target_log_q: (B, S_padded) log-normalized charges (ground truth)
            z: (B, latent_dim) - z[0] contains log_true_length
            attention_mask: (B, S_padded) boolean, True for padding tokens (False for valid tokens in target)
            true_sequence_lengths: (B,) int, true number of (t,q) data points in each sequence.
        Returns:
            mse_t: MSE for t
            mse_q: MSE for q
        """
        B, S_padded = target_log_t.shape
        device = target_log_t.device

        # Get full sequence predictions from the non-autoregressive decoder
        # _decode_non_autoregressive now takes z (which includes length info)
        # and returns (log_t_preds, log_q_preds) of shape (B, max_seq_len_padding, 2)
        predicted_log_tq = self._decode_non_autoregressive(z) # (B, max_seq_len_padding, 2)

        # Slice predictions to match batch's S_padded length for loss calculation
        pred_log_t_batch = predicted_log_tq[..., 0][:, :S_padded]  # (B, S_padded)
        pred_log_q_batch = predicted_log_tq[..., 1][:, :S_padded]  # (B, S_padded)

        # Mask for actual data points based on true_sequence_lengths
        # attention_mask is True for padding, so ~attention_mask is True for valid tokens in target_log_t/q
        # We only want to calculate loss on the actual data points, up to true_sequence_lengths
        data_points_mask = torch.zeros_like(target_log_t, dtype=torch.bool) # (B, S_padded)
        for i in range(B):
            actual_len = true_sequence_lengths[i].item()
            if actual_len > 0:
                data_points_mask[i, :actual_len] = True
        
        # Combine with original padding mask to ensure we don't go beyond padded length if error in true_sequence_lengths
        valid_loss_mask = data_points_mask & (~attention_mask)


        if valid_loss_mask.any():
            mse_t = F.mse_loss(pred_log_t_batch[valid_loss_mask], target_log_t[valid_loss_mask])
            mse_q = F.mse_loss(pred_log_q_batch[valid_loss_mask], target_log_q[valid_loss_mask])
        else:
            mse_t = torch.tensor(0.0, device=device, requires_grad=True) # Ensure gradients can flow if no valid tokens
            mse_q = torch.tensor(0.0, device=device, requires_grad=True)
            
        return mse_t, mse_q
    

    def _decode_non_autoregressive(self, z):
        """Internal helper for cross-attention decoding with memory bank."""
        B = z.size(0)
        device = z.device
        
        # z[:, 0] is log_true_length, z[:, 1:] is content_latent
        z_content = z[:, 1:] # (B, latent_dim - 1)

        # 1. Project z_content and expand to sequence length
        z_dec_input = self.decoder_z_proj(z_content) # (B, embed_dim)
        decoder_input_sequence = z_dec_input.unsqueeze(1).repeat(1, self.hparams.max_seq_len_padding, 1)
        
        # 2. Add positional encoding to decoder input (queries)
        decoder_input_sequence = self.decoder_pos_encoder(decoder_input_sequence) # (B, max_seq_len_padding, embed_dim)
        
        # 3. Prepare memory for cross-attention
        # Combine memory bank with latent information
        # Expand z_content to match memory bank format and concatenate
        z_expanded = z_dec_input.unsqueeze(1) # (B, 1, embed_dim)
        memory_bank_expanded = self.memory_bank.unsqueeze(0).expand(B, -1, -1) # (B, memory_bank_size, embed_dim)
        
        # Memory is concatenation of latent info + memory bank
        memory = torch.cat([z_expanded, memory_bank_expanded], dim=1) # (B, 1 + memory_bank_size, embed_dim)
        
        # 4. Pass through cross-attention decoder
        # tgt: decoder_input_sequence (queries)
        # memory: combined latent + memory bank (keys/values)
        transformed_sequence = self.decoder_transformer(
            tgt=decoder_input_sequence,  # (B, max_seq_len_padding, embed_dim)
            memory=memory               # (B, 1 + memory_bank_size, embed_dim)
        ) # (B, max_seq_len_padding, embed_dim)
        
        # 5. Project to output (log_t, log_q)
        predicted_log_tq = self.decoder_out_proj(transformed_sequence) # (B, max_seq_len_padding, 2)
        return predicted_log_tq

    def decode(self, z):
        """
        Non-autoregressively generate a sequence of (log t, log q) given latent z.
        Args:
            z: (B, latent_dim)
        Returns:
            generated_sequences: (B, L_i, 2) log-normalised (t, q) sequences, where L_i is determined by z[0].
                                 Padded to the max length among L_i in the batch.
        """
        B, device = z.size(0), z.device

        # Extract predicted lengths from z
        log_lengths = z[:, 0] # (B,)
        # Convert log-lengths to integer lengths for slicing
        # Ensure lengths are at least 1 and at most max_seq_len_padding
        predicted_lengths = torch.exp(log_lengths).round().long().clamp(min=1, max=self.hparams.max_seq_len_padding) # (B,)

        # Get full sequence predictions (log_t, log_q)
        # _decode_non_autoregressive now returns (B, max_seq_len_padding, 2)
        predicted_log_tq_full = self._decode_non_autoregressive(z)

        pred_log_t_full = predicted_log_tq_full[..., 0]  # (B, max_seq_len_padding)
        pred_log_q_full = predicted_log_tq_full[..., 1]  # (B, max_seq_len_padding)

        output_sequences = []
        max_len_batch = 0 # Determine max length in this batch for padding
        for i in range(B):
            # Use the length derived from z[0]
            length = predicted_lengths[i].item()
            
            # Slice the predictions to the determined length
            current_seq_log_t = pred_log_t_full[i, :length]
            current_seq_log_q = pred_log_q_full[i, :length]
            
            current_seq = torch.stack([current_seq_log_t, current_seq_log_q], dim=-1) # (length, 2)
            output_sequences.append(current_seq)
            if length > max_len_batch:
                max_len_batch = length
        
        if max_len_batch == 0: # Handle case where all sequences might be empty if lengths are 0 (though clamped to 1)
             max_len_batch = 1


        generated_padded = torch.zeros(B, max_len_batch, 2, device=device)
        for i, seq in enumerate(output_sequences):
            if seq.shape[0] > 0: # Ensure sequence is not empty before trying to assign
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
        attention_mask = batch['attention_mask'].bool() # True for padding
        # sequence_lengths from dataloader now directly represents the number of (t,q) pairs
        true_sequence_lengths = batch['sequence_lengths'].clamp(min=0)
        # target_eos = batch['eos_target_padded'].float() # Removed from batch

        # Beta annealing logic remains
        if not hasattr(self, "steps_initialized") or not self.steps_initialized:
            if self.trainer and hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader:
                steps_per_epoch = len(self.trainer.train_dataloader)
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch
            else: # Fallback if trainer or dataloader not available (e.g. direct call)
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * 100 # Arbitrary large number
            self.steps_initialized = True
        
        # Pass true_sequence_lengths (number of t,q pairs) to reconstruction_loss
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
        loss = reconstruction_loss + (self.beta * kl_loss)
        self.current_train_iter += 1
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("train_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        # self.log("train_eos_loss", eos_loss, batch_size=self.hparams.batch_size, sync_dist=True) # EOS loss removed
        self.log("beta", self.beta, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        true_sequence_lengths = batch['sequence_lengths'].clamp(min=0)
        # target_eos = batch['eos_target_padded'].float() # Removed


        mse_t, mse_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, true_sequence_lengths
        )

        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)

        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        # self.log("val_eos_loss", eos_loss, batch_size=self.hparams.batch_size, sync_dist=True) # EOS loss removed
        return loss
    
    def test_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()
        true_sequence_lengths = batch['sequence_lengths'].clamp(min=0)
        # target_eos = batch['eos_target_padded'].float() # Removed

        mse_t, mse_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, true_sequence_lengths
        )

        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)

        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        # self.log("test_eos_loss", eos_loss, batch_size=self.hparams.batch_size, sync_dist=True) # EOS loss removed
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[0], eta_min=self.hparams.lr_schedule[1])
        return [optimizer], [scheduler]