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
                 weight_decay=1e-5,
                 # --- Teacher Forcing Curriculum Hyperparameters ---
                 teacher_forcing_start_ratio=1.0,
                 teacher_forcing_end_ratio=0.0,
                 teacher_forcing_decay_type='cosine',
                 teacher_forcing_warmup_epochs=0,
                 teacher_forcing_decay_epochs=5
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

        # --- Teacher Forcing Curriculum State ---
        self.current_teacher_forcing_ratio = self.hparams.teacher_forcing_start_ratio
        self.total_steps_for_tf_warmup = 0
        self.total_steps_for_tf_decay = 0
        self.steps_initialized = False

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

    def reconstruction_loss(self, target_log_t, target_log_q, z, attention_mask, target_eos, sequence_lengths, teacher_forcing_ratio):
        """
        Args:
            target_log_t: (B, S) log-normalized arrival times
            target_log_q: (B, S) log-normalized charges
            z: (B, latent_dim)
            attention_mask: (B, S) boolean, True for padding tokens (False for valid)
            target_eos: (B, S) float, EOS targets (0 or 1)
            sequence_lengths: (B,) int, true sequence lengths (including EOS)
            teacher_forcing_ratio: float, probability of using teacher forcing at each step
        Returns:
            mse_t: MSE for t (masked, excluding EOS and padding)
            mse_q: MSE for q (masked, excluding EOS and padding)
            eos_loss: BCE loss for EOS prediction (masked, includes EOS)
        """
        B, S = target_log_t.shape
        device = target_log_t.device
        use_tf = torch.rand(B, device=device) < teacher_forcing_ratio  # (B,)
        # 1. First pass: all-teacher-forced input
        teacher_inputs = torch.zeros(B, S, 2, device=device)
        teacher_inputs[:, 1:, :] = torch.stack([target_log_t, target_log_q], dim=-1)[:, :-1, :]
        with torch.no_grad():
            teacher_logits = self.decoder(teacher_inputs, z, padding_mask=attention_mask)  # (B, S, 3)
            teacher_pred_log_t = teacher_logits[..., 0]
            teacher_pred_log_q = teacher_logits[..., 1]
        # 2. Prepare mixed input: for no-teacher sequences, use model's own predictions
        mixed_inputs = teacher_inputs.clone()
        # For each sequence in the batch, if not teacher-forced, fill in with model predictions
        for b in range(B):
            if not use_tf[b]:
                # Fill in positions 1:S with previous step's prediction (autoregressive)
                mixed_inputs[b, 1:, 0] = teacher_pred_log_t[b, :-1]
                mixed_inputs[b, 1:, 1] = teacher_pred_log_q[b, :-1]
        logits = self.decoder(mixed_inputs, z, padding_mask=attention_mask)  # (B, S, 3)
        pred_log_t = logits[..., 0]
        pred_log_q = logits[..., 1]
        pred_eos_logits = logits[..., 2]
        # NLL target padding mask: True for padding/EOS, False for valid data steps to include in NLL.
        nll_padding_mask = torch.ones_like(attention_mask, dtype=torch.bool)
        for i in range(B):
            actual_data_len = sequence_lengths[i].item() - 1
            if actual_data_len > 0:
                nll_padding_mask[i, :actual_data_len] = False
        valid_nll_indices = ~nll_padding_mask
        if valid_nll_indices.any():
            mse_t = F.mse_loss(pred_log_t[valid_nll_indices], target_log_t[valid_nll_indices])
            mse_q = F.mse_loss(pred_log_q[valid_nll_indices], target_log_q[valid_nll_indices])
        else:
            mse_t = torch.tensor(0.0, device=target_log_t.device, requires_grad=True)
            mse_q = torch.tensor(0.0, device=target_log_q.device, requires_grad=True)
        eos_loss = F.binary_cross_entropy_with_logits(
            pred_eos_logits[~attention_mask], target_eos[~attention_mask], reduction='mean'
        )
        return mse_t, mse_q, eos_loss
    

    def decode(self, z):
        """
        Auto‑regressively generate a sequence of (log t, log q) given latent z,
        sampling from the point estimate decoder distribution.
        Args:
            z: (B, latent_dim)
        Returns:
            generated: (B, L_i, 2)  log‑normalised (t, q) sequence, padded to max length
        """
        B, device = z.size(0), z.device
        max_steps      = self.hparams.max_seq_len_padding
        prev_tokens    = torch.zeros(B, 1, 2, device=device)  # SOS token: (log t, log q) = 0
        alive          = torch.ones(B, dtype=torch.bool, device=device)
        sequences      = [[] for _ in range(B)]

        for _ in range(max_steps):
            padding_mask = torch.zeros_like(prev_tokens[..., 0], dtype=torch.bool)
            decoder_out  = self.decoder(prev_tokens, z, padding_mask)       # (B, S_prev, 3)
            step_params  = decoder_out[:, -1, :]                            # last time‑step (B, 3)

            pred_log_t_step = step_params[:, 0]    # (B,)
            pred_log_q_step = step_params[:, 1]    # (B,)
            eos_logits      = step_params[:, 2]    # (B,)

            next_t = pred_log_t_step.unsqueeze(1)  # (B, 1)
            next_q = pred_log_q_step.unsqueeze(1)  # (B, 1)

            next_token   = torch.cat([next_t, next_q], dim=-1).unsqueeze(1) # (B, 1, 2)

            eos_prob     = torch.sigmoid(eos_logits)                        # (B,)
            eos_sample   = Bernoulli(probs=eos_prob).sample()               # (B,)

            for i in range(B):
                if alive[i]:
                    sequences[i].append(next_token[i, 0].detach())
                    if eos_sample[i] == 1:
                        alive[i] = False

            if not alive.any():        # all sequences ended
                break

            prev_tokens = torch.cat([prev_tokens, next_token], dim=1)      # grow sequence

        max_len = max(len(seq) for seq in sequences)
        generated = torch.zeros(B, max_len, 2, device=device)
        for i, seq in enumerate(sequences):
            if seq:
                generated[i, :len(seq), :] = torch.stack(seq)

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
        # --- Combined Initialization for Beta and Teacher Forcing Steps (runs once) ---
        if not self.steps_initialized:
            if self.trainer and hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader:
                steps_per_epoch = len(self.trainer.train_dataloader)
                self.total_steps_for_beta_annealing = self.hparams.beta_peak_epoch * steps_per_epoch
                self.total_steps_for_tf_warmup = self.hparams.teacher_forcing_warmup_epochs * steps_per_epoch
                self.total_steps_for_tf_decay = self.hparams.teacher_forcing_decay_epochs * steps_per_epoch
            else:
                self.total_steps_for_beta_annealing = 1
                self.total_steps_for_tf_warmup = self.hparams.teacher_forcing_warmup_epochs
                self.total_steps_for_tf_decay = self.hparams.teacher_forcing_decay_epochs if self.hparams.teacher_forcing_decay_epochs > 0 else 0
                if self.hparams.teacher_forcing_decay_epochs > 0 and self.total_steps_for_tf_decay == 0:
                    self.total_steps_for_tf_decay = 1
            self.steps_initialized = True
        # --- Teacher Forcing Ratio Update Logic (Iteration-based) ---
        if self.total_steps_for_tf_decay == 0:
            self.current_teacher_forcing_ratio = self.hparams.teacher_forcing_start_ratio
        elif self.current_train_iter < self.total_steps_for_tf_warmup:
            self.current_teacher_forcing_ratio = self.hparams.teacher_forcing_start_ratio
        elif self.current_train_iter < (self.total_steps_for_tf_warmup + self.total_steps_for_tf_decay):
            progress_in_decay = (self.current_train_iter - self.total_steps_for_tf_warmup) / float(self.total_steps_for_tf_decay)
            if self.hparams.teacher_forcing_decay_type == 'linear':
                self.current_teacher_forcing_ratio = self.hparams.teacher_forcing_start_ratio - \
                    progress_in_decay * (self.hparams.teacher_forcing_start_ratio - self.hparams.teacher_forcing_end_ratio)
            elif self.hparams.teacher_forcing_decay_type == 'cosine':
                self.current_teacher_forcing_ratio = (
                    self.hparams.teacher_forcing_end_ratio
                    + 0.5 * (self.hparams.teacher_forcing_start_ratio - self.hparams.teacher_forcing_end_ratio) * (1 + np.cos(np.pi * progress_in_decay))
                )
            self.current_teacher_forcing_ratio = np.clip(self.current_teacher_forcing_ratio, self.hparams.teacher_forcing_end_ratio, self.hparams.teacher_forcing_start_ratio)
        else:
            self.current_teacher_forcing_ratio = self.hparams.teacher_forcing_end_ratio
        self.log("teacher_forcing_ratio", self.current_teacher_forcing_ratio, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=True)
        mse_t, mse_q, eos_loss = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths,
            teacher_forcing_ratio=self.current_teacher_forcing_ratio
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
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths,
            teacher_forcing_ratio=1.0
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
            times_padded, counts_padded, z, attention_mask, target_eos, sequence_lengths,
            teacher_forcing_ratio=1.0
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
        # For point estimate: output log_t, log_q, eos_logit
        self.out_proj = nn.Linear(embed_dim, 3)

    def forward(self, tgt, z, padding_mask=None):
        # tgt: (B, S_tgt, 2) - current target sequence
        # z: (B, latent_dim) - latent vector
        # padding_mask: (B, S_tgt) - Optional, True for padding tokens, False for valid tokens
        
        B, S_tgt, _ = tgt.shape
        device = tgt.device

        # 1. Prepare target embeddings
        tgt_emb = self.input_proj(tgt)  # (B, S_tgt, embed_dim)
        tgt_emb = self.pos_encoder(tgt_emb)
        # Inject latent z into every decoder position
        z_proj = self.z_proj(z).unsqueeze(1)  # (B, 1, embed_dim)
        tgt_emb = tgt_emb + z_proj  # broadcast add to every position

        # 2. Prepare memory (context from z)
        # memory shape: (B, S_mem, embed_dim). Here S_mem = 1.
        memory = self.z_proj(z).unsqueeze(1) # (B, 1, embed_dim)

        # 3. Create Target Padding Mask (for nn.TransformerDecoder's tgt_key_padding_mask)
        # Shape: (B, S_tgt)
        tgt_key_padding_mask = None
        if padding_mask is not None:
            tgt_key_padding_mask = padding_mask
        
        causal_tgt_mask = nn.Transformer.generate_square_subsequent_mask(S_tgt, device=device).bool()

        # 4. Pass to the actual nn.TransformerDecoder
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=causal_tgt_mask,
            tgt_is_causal=True
        )
        return self.out_proj(out)