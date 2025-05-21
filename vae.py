import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import PositionalEncoding

class NT_VAE(pl.LightningModule):
    def __init__(self,
                 # in_features removed
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
        self.to_latent_mu = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)
        self.to_latent_logvar = nn.Linear(self.encoder_to_latent_input_dim, self.hparams.latent_dim)
        
        # Decoder: Auto-regressive Transformer Decoder for photon hits (log-normalized t, q, eos)
        self.decoder = Om2VecDecoder(
            latent_dim=self.hparams.latent_dim, # Now the full learned latent_dim
            embed_dim=self.hparams.embed_dim, # This is d_model for the decoder's internal workings
            num_layers=self.hparams.transformer_decoder_layers,
            num_heads=self.hparams.transformer_decoder_heads,
            ff_dim=self.hparams.transformer_decoder_ff_dim,
            dropout=self.hparams.transformer_decoder_dropout
        )

        self.beta = 0.
        self.current_train_iter = 0 # Renamed from self.iter for clarity

        # EOS loss weight hyperparameter
        self.hparams.eos_loss_weight = getattr(self.hparams, "eos_loss_weight", 1.0)

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
        Args:
            target_log_t: (B, S) log-normalized arrival times
            target_log_q: (B, S) log-normalized charges
            z: (B, latent_dim)
            attention_mask: (B, S) boolean, True for valid tokens
            target_eos: (B, S) float, EOS targets (0 or 1)
            sequence_lengths: (B,) int, true sequence lengths (including EOS)
        Returns:
            mse_t: MSE for t (masked, excluding EOS and padding)
            mse_q: MSE for q (masked, excluding EOS and padding)
            eos_loss: BCE loss for EOS prediction (masked, includes EOS)
        """
        B, S = target_log_t.shape
        device = target_log_t.device
        sos = torch.zeros((B, 1, 2), device=device)
        decoder_inputs = torch.cat([sos, torch.stack([target_log_t, target_log_q], dim=-1)[:, :-1, :]], dim=1)  # (B, S, 2)
        # Shift attention mask to align with decoder_inputs (teacher forcing)
        sos_mask_segment = torch.full_like(attention_mask[:, :1], True, dtype=torch.bool, device=attention_mask.device)
        main_mask_segment = attention_mask[:, :-1]
        shifted_attention_mask = torch.cat([sos_mask_segment, main_mask_segment], dim=1)
        # Forward through transformer decoder
        pred_outputs = self.decoder(decoder_inputs, z, padding_mask=shifted_attention_mask)  # (B, S, 3)
        pred_log_t = pred_outputs[..., 0]
        pred_log_q = pred_outputs[..., 1]
        pred_eos_logits = pred_outputs[..., 2]

        # Data step attention mask: True for indices 0 to sequence_lengths[i]-2 (excludes EOS and padding)
        data_step_attention_mask = torch.zeros_like(attention_mask)
        for i in range(B):
            seq_len = sequence_lengths[i].item()
            if seq_len > 1:
                data_step_attention_mask[i, :seq_len-1] = True

        mse_t = F.mse_loss(pred_log_t[data_step_attention_mask], target_log_t[data_step_attention_mask], reduction='mean')
        mse_q = F.mse_loss(pred_log_q[data_step_attention_mask], target_log_q[data_step_attention_mask], reduction='mean')

        # EOS loss: use original attention_mask (covers all valid steps including EOS)
        eos_loss = F.binary_cross_entropy_with_logits(
            pred_eos_logits[attention_mask], target_eos[attention_mask], reduction='mean'
        )
        return mse_t, mse_q, eos_loss
    
    def decode(self, z):
        """
        Auto-regressively generate a sequence of (log t, log q) given latent z, stopping at EOS.
        Args:
            z: (B, latent_dim)
        Returns:
            generated: (B, L_i, 2) log-normalized (t, q) sequence, padded to max generated length
        """
        B = z.shape[0]
        device = z.device
        max_steps = self.hparams.max_seq_len_padding
        prev = torch.zeros((B, 1, 2), device=device)
        active_sequences = torch.ones(B, dtype=torch.bool, device=device)
        generated_lists = [[] for _ in range(B)]
        for _ in range(max_steps):
            out = self.decoder(prev, z, padding_mask=None)  # (B, step+1, 3)
            next_pred_all = out[:, -1:, :]  # (B, 1, 3)
            next_token_tq_vals = next_pred_all[..., :2]  # (B, 1, 2)
            next_eos_logit = next_pred_all[..., 2].squeeze(-1)  # (B,)
            next_eos_prob = torch.sigmoid(next_eos_logit)  # (B,)

            for i in range(B):
                if active_sequences[i]:
                    generated_lists[i].append(next_token_tq_vals[i, 0].detach().clone())
                    if next_eos_prob[i] > 0.5:
                        active_sequences[i] = False

            if not active_sequences.any():
                break

            # Prepare next input: only update for active sequences
            next_input = prev.clone()
            for i in range(B):
                if active_sequences[i]:
                    next_input[i] = torch.cat([prev[i], next_token_tq_vals[i]], dim=0)
            prev = next_input

        # Pad generated sequences to max length in batch
        max_len = max(len(seq) for seq in generated_lists)
        generated = torch.zeros((B, max_len, 2), device=device)
        for i, seq in enumerate(generated_lists):
            if len(seq) > 0:
                generated[i, :len(seq), :] = torch.stack(seq, dim=0)
        return generated
    
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

        mse_t, mse_q, eos_loss = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths
        )

        reconstruction_loss_time = mse_t
        reconstruction_loss_charge = mse_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge + self.hparams.eos_loss_weight * eos_loss

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
    
# --- Om2VecDecoder module ---
class Om2VecDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_layers=4, num_heads=8, ff_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.z_proj = nn.Linear(latent_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, 3)  # Predict mean log(t), log(q), eos_logit

    def forward(self, tgt, z, padding_mask=None):
        # tgt: (B, S_tgt, 2) - current target sequence
        # z: (B, latent_dim) - latent vector
        # padding_mask: (B, S_tgt) - Optional, True for VALID tokens, False for PADDED tokens
        
        B, S_tgt, _ = tgt.shape
        device = tgt.device

        # 1. Prepare target embeddings
        tgt_emb = self.input_proj(tgt)  # (B, S_tgt, embed_dim)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 2. Prepare memory (context from z)
        # memory shape: (B, S_mem, embed_dim). Here S_mem = 1.
        memory = self.z_proj(z).unsqueeze(1) # (B, 1, embed_dim)

        # 3. Create Target Padding Mask (for nn.TransformerDecoder's tgt_key_padding_mask)
        # Shape: (B, S_tgt)
        tgt_key_padding_mask = None
        if padding_mask is not None:
            tgt_key_padding_mask = ~padding_mask

        causal_tgt_mask = nn.Transformer.generate_square_subsequent_mask(S_tgt, device=device)

        # 4. Pass to the actual nn.TransformerDecoder
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=causal_tgt_mask,
            tgt_is_causal=True
        )
        return self.out_proj(out)