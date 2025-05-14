import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import LogNormal, Normal
# Ensure utils.utils can be imported. If run.py adds project root to sys.path, this should work.
from utils.utils import log_transform, inverse_log_transform

class Om2vecModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # Save the entire config dictionary as hyperparameters.
        # This makes all config values accessible via self.hparams.
        self.save_hyperparameters(cfg)
        # It's also common to store cfg directly if preferred for direct access,
        # but self.hparams is the PyTorch Lightning way.
        # self.cfg = cfg 

        model_options = self.hparams['model_options']
        data_options = self.hparams['data_options'] # For max_seq_len

        # ---- Encoder ----
        self.encoder_input_embed_dim_time = model_options['encoder_input_embed_dim_time']
        self.encoder_input_embed_dim_charge = model_options['encoder_input_embed_dim_charge']
        combined_embed_dim_enc = self.encoder_input_embed_dim_time + self.encoder_input_embed_dim_charge

        self.time_embed_enc = nn.Linear(1, self.encoder_input_embed_dim_time)
        self.charge_embed_enc = nn.Linear(1, self.encoder_input_embed_dim_charge)

        # Hardcoding to LSTM as per user feedback
        self.encoder_rnn = nn.LSTM(
            input_size=combined_embed_dim_enc,
            hidden_size=model_options['encoder_hidden_dim'],
            num_layers=model_options['encoder_num_layers'],
            batch_first=True
        )
        
        # Latent dim for z[1:] (learned part)
        self.learned_latent_dim = model_options['latent_dim'] - 1
        if self.learned_latent_dim <= 0:
            raise ValueError("model_options['latent_dim'] must be > 1 to accommodate the learned part.")

        self.fc_mu_rest = nn.Linear(model_options['encoder_hidden_dim'], self.learned_latent_dim)
        self.fc_logvar_rest = nn.Linear(model_options['encoder_hidden_dim'], self.learned_latent_dim)

        # ---- Decoder (NTPP) ----
        self.latent_to_decoder_hidden = nn.Linear(model_options['latent_dim'], model_options['decoder_hidden_dim'])
        
        # Decoder RNN input: previous (embedded_delta_t, embedded_charge)
        # Using same embedding dimensions for decoder as encoder for simplicity
        self.decoder_input_embed_dim_time = model_options['encoder_input_embed_dim_time']
        self.decoder_input_embed_dim_charge = model_options['encoder_input_embed_dim_charge']
        combined_embed_dim_dec = self.decoder_input_embed_dim_time + self.decoder_input_embed_dim_charge

        self.time_embed_dec = nn.Linear(1, self.decoder_input_embed_dim_time)
        self.charge_embed_dec = nn.Linear(1, self.decoder_input_embed_dim_charge)

        # Hardcoding to LSTM as per user feedback
        self.decoder_rnn = nn.LSTM(
            input_size=combined_embed_dim_dec,
            hidden_size=model_options['decoder_hidden_dim'],
            num_layers=model_options['decoder_num_layers'],
            batch_first=True
        )

        # Output layers for NTPP parameters
        self.fc_delta_t_loc = nn.Linear(model_options['decoder_hidden_dim'], 1)
        self.fc_delta_t_scale = nn.Linear(model_options['decoder_hidden_dim'], 1)
        self.fc_charge_loc = nn.Linear(model_options['decoder_hidden_dim'], 1)
        self.fc_charge_scale = nn.Linear(model_options['decoder_hidden_dim'], 1)

        self.max_seq_len = data_options['max_seq_len']

        # KL Annealing parameters
        self.kl_beta_start = model_options.get('kl_beta_start', 1.0) # Default to 1.0 if not specified
        self.kl_beta_end = model_options.get('kl_beta_end', 1.0)
        self.kl_anneal_epochs = model_options.get('kl_anneal_epochs', 0) # Default to 0 epochs (no annealing)

    def encode(self, input_tq_sequence, attention_mask):
        # input_tq_sequence: (batch, seq_len, 2) [norm_abs_t, norm_abs_q]
        norm_t = input_tq_sequence[:, :, 0:1]
        norm_q = input_tq_sequence[:, :, 1:2]

        embedded_t = torch.relu(self.time_embed_enc(norm_t))
        embedded_q = torch.relu(self.charge_embed_enc(norm_q))
        encoder_input = torch.cat([embedded_t, embedded_q], dim=2)

        # Using attention_mask to get actual lengths for packing
        # Ensure sequence_lengths are on CPU for pack_padded_sequence if it requires it,
        # but typically lengths should be on the same device as the input tensor.
        # The .sum() will be on the same device as attention_mask.
        # Ensure lengths are > 0. If a sequence has 0 length, it might cause issues.
        # The dataloader should ensure active_entries_count >= 1, which implies attention_mask.sum() >= 1.
        sequence_lengths = attention_mask.sum(dim=1).long()
        
        # Handle cases where a sequence might have zero length if attention_mask is all False.
        # Clamp lengths to be at least 1 for packing, as pack_padded_sequence expects positive lengths.
        # This scenario (all-False mask) should ideally be prevented by the dataloader ensuring min 1 active entry.
        clamped_lengths = torch.clamp(sequence_lengths, min=1)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            encoder_input, clamped_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # packed_output is a PackedSequence object, hidden_state contains final states
        packed_output, hidden_state = self.encoder_rnn(packed_input)
        
        # hidden_state for LSTM is a tuple (h_n, c_n)
        # h_n is of shape (num_layers * num_directions, batch, hidden_size)
        # We want the final hidden state of the last layer for each sequence in the batch.
        # For batch_first=True, hidden_state[0] is h_n.
        # hidden_state[0][-1] gives the last layer's h_n for all sequences.
        final_hidden = hidden_state[0][-1] # (batch, hidden_size)

        mu_rest = self.fc_mu_rest(final_hidden)
        logvar_rest = self.fc_logvar_rest(final_hidden)
        return mu_rest, logvar_rest

    def reparameterize(self, mu_rest, logvar_rest):
        std_rest = torch.exp(0.5 * logvar_rest)
        eps_rest = torch.randn_like(std_rest)
        return mu_rest + eps_rest * std_rest

    def decode(self, z, ground_truth_sequence_for_teacher_forcing=None):
        # z: (batch, total_latent_dim)
        # ground_truth_sequence_for_teacher_forcing: (batch, seq_len, 2) [norm_delta_t, norm_charge]
        
        batch_size = z.size(0)
        
        # Initial hidden state for decoder RNN from full z
        decoder_h0 = self.latent_to_decoder_hidden(z)
        decoder_h0 = decoder_h0.unsqueeze(0).repeat(self.hparams['model_options']['decoder_num_layers'], 1, 1)
        # self.decoder_rnn is now always nn.LSTM
        decoder_hidden = (decoder_h0, torch.zeros_like(decoder_h0, device=z.device)) # (h_0, c_0)

        # Initial input for the first step (start token)
        current_delta_t_input = torch.zeros(batch_size, 1, 1, device=z.device)
        current_charge_input = torch.zeros(batch_size, 1, 1, device=z.device)

        all_dt_locs, all_dt_scales = [], []
        all_charge_locs, all_charge_scales = [], []

        for k in range(self.max_seq_len): # Decoder runs for max_seq_len for training
            embedded_dt = torch.relu(self.time_embed_dec(current_delta_t_input))
            embedded_q = torch.relu(self.charge_embed_dec(current_charge_input))
            decoder_input_k = torch.cat([embedded_dt, embedded_q], dim=2)

            rnn_output_k, decoder_hidden = self.decoder_rnn(decoder_input_k, decoder_hidden)
            
            dt_loc = self.fc_delta_t_loc(rnn_output_k)
            dt_scale = torch.exp(self.fc_delta_t_scale(rnn_output_k)) # Ensure scale is positive
            charge_loc = self.fc_charge_loc(rnn_output_k)
            charge_scale = torch.exp(self.fc_charge_scale(rnn_output_k))

            all_dt_locs.append(dt_loc)
            all_dt_scales.append(dt_scale)
            all_charge_locs.append(charge_loc)
            all_charge_scales.append(charge_scale)

            if ground_truth_sequence_for_teacher_forcing is not None: # Teacher forcing
                current_delta_t_input = ground_truth_sequence_for_teacher_forcing[:, k:k+1, 0:1]
                current_charge_input = ground_truth_sequence_for_teacher_forcing[:, k:k+1, 1:2]
            else: # Inference/Generation (not used in ELBO calculation directly)
                # Sample from LogNormal(dt_loc, dt_scale) and Normal(charge_loc, charge_scale)
                # This logic is for the .generate() method
                dist_dt_k = LogNormal(dt_loc, dt_scale.clamp(min=1e-6))
                current_delta_t_input = dist_dt_k.sample()
                dist_charge_k = Normal(charge_loc, charge_scale.clamp(min=1e-6))
                current_charge_input = dist_charge_k.sample()


        return (torch.cat(all_dt_locs, dim=1), torch.cat(all_dt_scales, dim=1),
                torch.cat(all_charge_locs, dim=1), torch.cat(all_charge_scales, dim=1))

    def forward(self, batch):
        input_tq_sequence = batch["input_tq_sequence"] # (B, L, 2) [abs_norm_t, abs_norm_q]
        attention_mask = batch["attention_mask"]       # (B, L)
        active_entries_count = batch["active_entries_count"].float() # (B,)

        # z[0] component: log-normalized true length
        z0_true_log_len = torch.log(active_entries_count + 1.0).unsqueeze(1) # (B, 1)

        mu_rest, logvar_rest = self.encode(input_tq_sequence, attention_mask)
        z_rest = self.reparameterize(mu_rest, logvar_rest) # (B, learned_latent_dim)
        
        z = torch.cat([z0_true_log_len, z_rest], dim=1) # (B, total_latent_dim)

        # Prepare target for decoder: (norm_delta_t, norm_charge)
        abs_norm_t = input_tq_sequence[:, :, 0:1]
        norm_q = input_tq_sequence[:, :, 1:2] # This is already the charge per bin

        delta_norm_t = torch.zeros_like(abs_norm_t)
        delta_norm_t[:, 0, :] = abs_norm_t[:, 0, :] # delta_t_1 = t_1 (assuming t_0 = 0)
        if self.max_seq_len > 1: # Avoid issues if max_seq_len is 1
            delta_norm_t[:, 1:, :] = abs_norm_t[:, 1:, :] - abs_norm_t[:, :-1, :]
        
        # Mask out deltas where the time difference is due to padding (t_k is 0 and t_{k-1} was also 0)
        # This can be complex. A simpler way is to ensure that padded times are large negative or use the loss mask.
        # The loss mask will handle not penalizing these.

        decoder_target_sequence = torch.cat([delta_norm_t, norm_q], dim=2)

        dt_locs, dt_scales, charge_locs, charge_scales = self.decode(z, decoder_target_sequence)
        
        return dt_locs, dt_scales, charge_locs, charge_scales, mu_rest, logvar_rest, decoder_target_sequence, z0_true_log_len

    def _shared_step(self, batch, batch_idx, stage_name):
        dt_locs, dt_scales, charge_locs, charge_scales, mu_rest, logvar_rest, target_sequence, z0_true_log_len = self.forward(batch)
        
        target_delta_t = target_sequence[:, :, 0:1]
        target_charge = target_sequence[:, :, 1:2]

        # Determine actual lengths for loss masking from z0_true_log_len
        true_lengths_for_loss = torch.round(torch.exp(z0_true_log_len.squeeze(dim=-1)) - 1.0).long()
        true_lengths_for_loss = torch.clamp(true_lengths_for_loss, min=1, max=self.max_seq_len)

        loss_mask_seq = torch.arange(self.max_seq_len, device=target_sequence.device)[None, :] < true_lengths_for_loss[:, None]
        loss_mask = loss_mask_seq.unsqueeze(-1) # (B, L, 1) to broadcast with (B, L, 1) log_probs

        # Reconstruction Loss (Negative Log Likelihood)
        # For delta_t (LogNormal) - ensure target_delta_t is positive for log_prob
        dist_dt = LogNormal(dt_locs, dt_scales.clamp(min=1e-6))
        log_prob_dt = dist_dt.log_prob(target_delta_t.clamp(min=1e-6)) 
        
        # For charge (Normal)
        dist_charge = Normal(charge_locs, charge_scales.clamp(min=1e-6))
        log_prob_charge = dist_charge.log_prob(target_charge)

        masked_log_prob_dt = log_prob_dt * loss_mask
        masked_log_prob_charge = log_prob_charge * loss_mask

        # Sum over seq_len and feature_dim (1), then mean over batch dimension
        # For NLL, we want to maximize log_prob, so minimize -log_prob
        # Sum over active entries only. Divide by number of active entries in batch for mean.
        active_elements_per_batch = loss_mask.sum()
        
        recon_loss_dt = -masked_log_prob_dt.sum() / active_elements_per_batch.clamp(min=1)
        recon_loss_charge = -masked_log_prob_charge.sum() / active_elements_per_batch.clamp(min=1)
        reconstruction_loss = recon_loss_dt + recon_loss_charge

        # KL Divergence for z[1:] (learned part)
        kl_divergence = -0.5 * torch.sum(1 + logvar_rest - mu_rest.pow(2) - logvar_rest.exp(), dim=1).mean()
        
        # Calculate KL beta for annealing
        if self.kl_anneal_epochs > 0 and self.current_epoch < self.kl_anneal_epochs:
            kl_beta = self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * (self.current_epoch / self.kl_anneal_epochs)
        elif self.kl_anneal_epochs > 0 and self.current_epoch >= self.kl_anneal_epochs:
            kl_beta = self.kl_beta_end
        else: # No annealing or annealing finished
            kl_beta = self.kl_beta_end # Use end value (could be 1.0 if not specified for annealing)

        loss = reconstruction_loss + kl_beta * kl_divergence

        self.log(f"{stage_name}_recon_loss", reconstruction_loss, on_step=(stage_name=="train"), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage_name}_kl_div_raw", kl_divergence, on_step=(stage_name=="train"), on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage_name}_kl_beta", kl_beta, on_step=(stage_name=="train"), on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage_name}_kl_div_scaled", kl_beta * kl_divergence, on_step=(stage_name=="train"), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage_name}_loss", loss, on_step=(stage_name=="train"), on_epoch=True, prog_bar=True, logger=True)
        if stage_name == "train":  
            self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
                
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        opt_cfg = self.hparams['training_options']
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay']
        )
        # Example of adding a scheduler from config if present
        if "lr_schedule" in opt_cfg and opt_cfg["lr_schedule"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opt_cfg["lr_schedule"][0], eta_min=opt_cfg["lr_schedule"][1]
            )
            return [optimizer], [scheduler]

    @torch.no_grad()
    def generate(self, num_samples: int, z_vectors_provided: torch.Tensor, device: torch.device):
        # z_vectors_provided: (num_samples, total_latent_dim), where z_vectors_provided[:, 0] is log(N_gen + 1)
        self.eval()
        
        all_generated_sequences_t_norm = []
        all_generated_sequences_q_norm = []

        for i in range(num_samples):
            z_sample_i = z_vectors_provided[i:i+1, :].to(device) # (1, total_latent_dim)

            n_steps_for_this_sample = torch.round(torch.exp(z_sample_i[:, 0]) - 1.0).int().item()
            n_steps_for_this_sample = max(1, min(n_steps_for_this_sample, self.max_seq_len))

            # Initial hidden state for decoder RNN from z_sample_i
            decoder_h0_gen = self.latent_to_decoder_hidden(z_sample_i)
            decoder_h0_gen = decoder_h0_gen.unsqueeze(0).repeat(self.hparams['model_options']['decoder_num_layers'], 1, 1)
            # self.decoder_rnn is now always nn.LSTM
            decoder_hidden_gen = (decoder_h0_gen, torch.zeros_like(decoder_h0_gen, device=device))
            
            current_delta_t_input = torch.zeros(1, 1, 1, device=device) # Start token
            current_charge_input = torch.zeros(1, 1, 1, device=device)  # Start token

            generated_sequence_t_single_norm = []
            generated_sequence_q_single_norm = []

            for _ in range(n_steps_for_this_sample):
                embedded_dt = torch.relu(self.time_embed_dec(current_delta_t_input))
                embedded_q = torch.relu(self.charge_embed_dec(current_charge_input))
                decoder_input_k = torch.cat([embedded_dt, embedded_q], dim=2)

                rnn_output_k, decoder_hidden_gen = self.decoder_rnn(decoder_input_k, decoder_hidden_gen)

                dt_loc = self.fc_delta_t_loc(rnn_output_k)
                dt_scale = torch.exp(self.fc_delta_t_scale(rnn_output_k))
                charge_loc = self.fc_charge_loc(rnn_output_k)
                charge_scale = torch.exp(self.fc_charge_scale(rnn_output_k))

                next_delta_t_norm = LogNormal(dt_loc, dt_scale.clamp(min=1e-6)).sample()
                next_charge_norm = Normal(charge_loc, charge_scale.clamp(min=1e-6)).sample()
                
                generated_sequence_t_single_norm.append(next_delta_t_norm.squeeze())
                generated_sequence_q_single_norm.append(next_charge_norm.squeeze())

                current_delta_t_input = next_delta_t_norm
                current_charge_input = next_charge_norm
            
            if generated_sequence_t_single_norm:
                all_generated_sequences_t_norm.append(torch.stack(generated_sequence_t_single_norm))
                all_generated_sequences_q_norm.append(torch.stack(generated_sequence_q_single_norm))
            else: # Should not happen if n_steps_for_this_sample >= 1
                all_generated_sequences_t_norm.append(torch.tensor([], device=device))
                all_generated_sequences_q_norm.append(torch.tensor([], device=device))

        # Return lists of normalized sequences. Unnormalization and conversion to absolute times
        # would typically be done by the caller or a post-processing utility.
        return all_generated_sequences_t_norm, all_generated_sequences_q_norm