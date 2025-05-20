import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import PositionalEncoding
from utils import mdn_negative_log_likelihood, point_estimate_mse_loss

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
                 decoder_type='mdn',
                 mdn_num_components=5,
                 batch_size=32,
                 lr=1e-3,
                 lr_schedule=[20, 1e-6],
                 weight_decay=1e-5,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.num_summary_stats = 10 # Fixed number of summary statistics
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
        
        # Decoder: Auto-regressive Transformer Decoder for photon hits (log-normalized t, q)
        self.decoder = Om2VecDecoder(
            latent_dim=self.hparams.latent_dim, # This is the full latent_dim (summary_stats + learned)
            embed_dim=self.hparams.embed_dim, # This is d_model for the decoder's internal workings
            num_layers=self.hparams.transformer_decoder_layers,
            num_heads=self.hparams.transformer_decoder_heads,
            ff_dim=self.hparams.transformer_decoder_ff_dim,
            dropout=self.hparams.transformer_decoder_dropout,
            num_mixture_components=self.hparams.mdn_num_components,
            decoder_type=self.hparams.decoder_type # Pass decoder_type to Om2VecDecoder
        )

        self.beta = 0.
        self.current_train_iter = 0 # Renamed from self.iter for clarity
        
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

        # clamp logvar_learned for stability
        logvar_learned = torch.clamp(logvar_learned, min=-10, max=10)

        return mu_learned, logvar_learned
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, max_len=None):
        """
        Auto-regressively generate a sequence of (log t, log q) given latent z.
        Args:
            z: (B, latent_dim)
            max_len: Optional[int], if None, use exp(z[0]) as sequence length (rounded)
        Returns:
            generated: (B, L, 2) log-normalized (t, q) sequence
        """
        B = z.shape[0]
        device = z.device
        if max_len is None:
            # z[:, 0] is log-normalized real sequence length
            lengths = torch.exp(z[:, 0]).round().long().clamp(min=1, max=self.hparams.max_seq_len_padding)
        else:
            lengths = torch.full((B,), max_len, dtype=torch.long, device=device)
        L = lengths.max().item()
        generated = []
        prev = torch.zeros((B, 1, 2), device=device)
        for _ in range(L):
            decoder_output = self.decoder(prev, z, padding_mask=None)

            if self.hparams.decoder_type == 'mdn':
                # Decoder returns GMM params: (pis_t, mus_t, sigmas_t, pis_q, mus_q, sigmas_q)
                pis_t, mus_t, sigmas_t, pis_q, mus_q, sigmas_q = decoder_output
                # Get the last predicted token's GMM params (B, 1, K)
                pis_t_last = pis_t[:, -1, :]  # (B, K)
                mus_t_last = mus_t[:, -1, :]  # (B, K)
                sigmas_t_last = sigmas_t[:, -1, :]  # (B, K)
                pis_q_last = pis_q[:, -1, :]  # (B, K)
                mus_q_last = mus_q[:, -1, :]  # (B, K)
                sigmas_q_last = sigmas_q[:, -1, :]  # (B, K)

                # Sample mixture component for each batch item
                k_t = torch.multinomial(pis_t_last, 1).squeeze(-1)  # (B,)
                k_q = torch.multinomial(pis_q_last, 1).squeeze(-1)  # (B,)

                # Gather means and stds for selected components
                mu_t = mus_t_last.gather(1, k_t.unsqueeze(1)).squeeze(1)      # (B,)
                sigma_t = sigmas_t_last.gather(1, k_t.unsqueeze(1)).squeeze(1)  # (B,)
                mu_q = mus_q_last.gather(1, k_q.unsqueeze(1)).squeeze(1)      # (B,)
                sigma_q = sigmas_q_last.gather(1, k_q.unsqueeze(1)).squeeze(1)  # (B,)

                # Sample from Normal for log(t) and log(q)
                log_t_val = torch.distributions.Normal(mu_t, sigma_t).sample()  # (B,)
                log_q_val = torch.distributions.Normal(mu_q, sigma_q).sample()  # (B,)

            elif self.hparams.decoder_type == 'point_estimate':
                # Decoder returns point estimates: (pred_log_t, pred_log_q)
                pred_log_t_steps, pred_log_q_steps = decoder_output
                log_t_val = pred_log_t_steps[:, -1] # (B,)
                log_q_val = pred_log_q_steps[:, -1] # (B,)
            else:
                raise ValueError(f"Unknown decoder_type for sampling: {self.hparams.decoder_type}")

            # Stack to get next_token (B, 2), then unsqueeze to (B, 1, 2)
            next_token = torch.stack((log_t_val, log_q_val), dim=-1).unsqueeze(1)  # (B, 1, 2)
            generated.append(next_token)
            prev = torch.cat([prev, next_token], dim=1)

        generated = torch.cat(generated, dim=1)  # (B, L, 2)
        # Optionally mask to true lengths
        for i, l in enumerate(lengths):
            if l < L:
                generated[i, l:] = 0
        return generated
    
    def reconstruction_loss(self, target_log_t, target_log_q, z, attention_mask):
        """
        Args:
            target_log_t: (B, S) log-normalized arrival times
            target_log_q: (B, S) log-normalized charges
            z: (B, latent_dim)
            attention_mask: (B, S) boolean, True for valid tokens
        Returns:
            nll_t: scalar NLL for t
            nll_q: scalar NLL for q
        """
        B, S = target_log_t.shape
        device = target_log_t.device
        sos = torch.zeros((B, 1, 2), device=device)
        decoder_inputs = torch.cat([sos, torch.stack([target_log_t, target_log_q], dim=-1)[:, :-1, :]], dim=1)  # (B, S, 2)
        decoder_output = self.decoder(decoder_inputs, z, padding_mask=attention_mask)

        if self.hparams.decoder_type == 'mdn':
            # Decoder returns GMM parameters
            pis_t, mus_t, sigmas_t, pis_q, mus_q, sigmas_q = decoder_output
            loss_t = mdn_negative_log_likelihood(target_log_t, pis_t, mus_t, sigmas_t, attention_mask)
            loss_q = mdn_negative_log_likelihood(target_log_q, pis_q, mus_q, sigmas_q, attention_mask)
        elif self.hparams.decoder_type == 'point_estimate':
            # Decoder returns point estimates
            pred_log_t, pred_log_q = decoder_output
            loss_t = point_estimate_mse_loss(target_log_t, pred_log_t, attention_mask)
            loss_q = point_estimate_mse_loss(target_log_q, pred_log_q, attention_mask)
        else:
            raise ValueError(f"Unknown decoder_type for reconstruction_loss: {self.hparams.decoder_type}")

        return loss_t, loss_q
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
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

    def training_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)
            
        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()

        nll_t, nll_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask
        )

        reconstruction_loss_time = nll_t
        reconstruction_loss_charge = nll_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge

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

        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()

        nll_t, nll_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask
        )

        reconstruction_loss_time = nll_t
        reconstruction_loss_charge = nll_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)

        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True, prog_bar=True)
        self.log("val_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        mu, logvar, z = self(batch)

        times_padded = batch['times_padded'].float()
        counts_padded = batch['counts_padded'].float()
        attention_mask = batch['attention_mask'].bool()

        nll_t, nll_q = self.reconstruction_loss(
            times_padded, counts_padded, z, attention_mask
        )

        reconstruction_loss_time = nll_t
        reconstruction_loss_charge = nll_q
        reconstruction_loss = reconstruction_loss_time + reconstruction_loss_charge

        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)

        self.log("test_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_kl_loss", kl_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_total", reconstruction_loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_time", reconstruction_loss_time, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test_reco_loss_charge", reconstruction_loss_charge, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[0], eta_min=self.hparams.lr_schedule[1])
        return [optimizer], [scheduler]
    
# --- Om2VecDecoder module ---
class Om2VecDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_layers=4, num_heads=8, ff_dim=256, dropout=0.1, decoder_type='mdn', num_mixture_components=5):
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
        self.num_mixture_components = num_mixture_components
        self.decoder_type = decoder_type
        if self.decoder_type == 'mdn':
            # Output: for each of log(t) and log(q): num_mixture_components * (pi, mu, sigma)
            self.out_proj = nn.Linear(embed_dim, num_mixture_components * 3 * 2)
        elif self.decoder_type == 'point_estimate':
            self.out_proj = nn.Linear(embed_dim, 2)
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

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

        # 3. Create Causal Mask for target self-attention (for nn.TransformerDecoder's tgt_mask)
        # Shape: (S_tgt, S_tgt)
        causal_tgt_mask = nn.Transformer.generate_square_subsequent_mask(S_tgt, device=device)

        # 4. Create Target Padding Mask (for nn.TransformerDecoder's tgt_key_padding_mask)
        # Shape: (B, S_tgt)
        tgt_key_padding_mask = None
        if padding_mask is not None:
            tgt_key_padding_mask = ~padding_mask

        # 5. Create Memory Padding Mask (for nn.TransformerDecoder's memory_key_padding_mask)
        # Shape: (B, S_mem) -> (B, 1)
        memory_key_padding_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        
        # 6. Pass to the actual nn.TransformerDecoder
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # out: (B, S_tgt, embed_dim)
        raw_out = self.out_proj(out)
        if self.decoder_type == 'mdn':
            n = self.num_mixture_components
            # Reshape to (B, S_tgt, 2, n, 3): 2 features, n components, 3 params (pi, mu, sigma)
            raw_out = raw_out.view(B, S_tgt, 2, n, 3)

            # Split for log(t) and log(q)
            raw_t = raw_out[:, :, 0, :, :]  # (B, S_tgt, n, 3)
            raw_q = raw_out[:, :, 1, :, :]  # (B, S_tgt, n, 3)

            # For each: [:, :, :, 0]=pi, [:, :, :, 1]=mu, [:, :, :, 2]=sigma
            pis_t = torch.softmax(raw_t[..., 0], dim=-1)  # (B, S_tgt, n)
            mus_t = raw_t[..., 1]                        # (B, S_tgt, n)
            sigmas_t = torch.nn.functional.softplus(raw_t[..., 2]) + 1e-4  # (B, S_tgt, n)

            pis_q = torch.softmax(raw_q[..., 0], dim=-1)  # (B, S_tgt, n)
            mus_q = raw_q[..., 1]                         # (B, S_tgt, n)
            sigmas_q = torch.nn.functional.softplus(raw_q[..., 2]) + 1e-4  # (B, S_tgt, n)

            return pis_t, mus_t, sigmas_t, pis_q, mus_q, sigmas_q

        elif self.decoder_type == 'point_estimate':
            # Output shape: (B, S_tgt, 2)
            pred_log_t = raw_out[..., 0]  # (B, S_tgt)
            pred_log_q = raw_out[..., 1]  # (B, S_tgt)
            return pred_log_t, pred_log_q

        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")