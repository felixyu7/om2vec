import torch
import torch.nn as nn
import pytorch_lightning as pl
import zuko
from torch.distributions import Normal, kl_divergence as kl

# Assuming utils.py is in ../utils/ relative to this file's eventual location
# For robust imports, run.py adds project root to sys.path
try:
    from utils import utils as project_utils
except ImportError:
    try:
        import utils as project_utils
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils import utils as project_utils


class FiLMLayer(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) Layer.
    Takes a conditioning vector (e.g., sensor position) and outputs
    gamma and beta parameters to modulate another feature vector.
    """
    def __init__(self, condition_dim, modulated_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (condition_dim + modulated_dim * 2) // 2 # A heuristic
        
        self.network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, modulated_dim * 2) # Output for gamma and beta
        )
        self.modulated_dim = modulated_dim

    def forward(self, condition_vector):
        # condition_vector shape: (batch, condition_dim)
        params = self.network(condition_vector) # (batch, modulated_dim * 2)
        gamma = params[:, :self.modulated_dim]
        beta = params[:, self.modulated_dim:]
        return gamma, beta


class Om2vecModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg) # Saves cfg to self.hparams
        
        # Accessible via self.hparams after save_hyperparameters()
        # e.g., self.hparams.model_options.latent_dim_learned

        # Import utility functions (or they can be passed/set if preferred)
        self.summary_stats_calculator = project_utils.calculate_summary_statistics
        self.normalize_tq = project_utils.normalize_tq
        self.denormalize_tq = project_utils.denormalize_tq
        self.normalize_sensor_pos = project_utils.normalize_sensor_pos
        self.denormalize_sensor_pos = project_utils.denormalize_sensor_pos
        self.normalize_z_summary = project_utils.normalize_z_summary
        self.denormalize_z_summary = project_utils.denormalize_z_summary
        
        # Accessing hparams with dictionary style for safety if cfg was a dict
        data_opts = self.hparams.get('data_options', {})
        model_opts = self.hparams.get('model_options', {})

        self.max_seq_len = data_opts.get('max_seq_len', 512)
        self.latent_dim_learned = model_opts.get('latent_dim_learned', 64)
        self.summary_stats_dim = data_opts.get('summary_stats_dim', 10) # Should be 10

        # --- Encoder Parts ---
        transformer_d_model = model_opts.get('transformer_d_model', 256) # Get with default
        self.input_tq_embedding = nn.Linear(2, transformer_d_model)
        
        self.positional_encoding = project_utils.PositionalEncodingBatchFirst(
            d_model=transformer_d_model,
            dropout=model_opts.get('transformer_dropout', 0.1),
            max_len=self.max_seq_len
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=model_opts.get('transformer_nhead', 8),
            dim_feedforward=model_opts.get('transformer_dim_feedforward', 1024),
            dropout=model_opts.get('transformer_dropout', 0.1),
            activation=model_opts.get('transformer_activation', 'relu'),
            batch_first=True,
            norm_first=model_opts.get('transformer_norm_first', False)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_opts.get('transformer_num_encoder_layers', 6)
        )
        
        self.latent_projection_head = nn.Linear(
            transformer_d_model,
            2 * self.latent_dim_learned
        )

        # --- FiLM Layer for Sensor Position ---
        self.film_layer = FiLMLayer(
            condition_dim=3, # sensor_pos_dim
            modulated_dim=self.latent_dim_learned,
            # hidden_dim can be specified in model_opts if needed, e.g., model_opts.get('film_hidden_dim', 64)
        )

        # --- Decoder Parts (CNF using zuko) ---
        # Sensor position is now incorporated via FiLM, so not directly in CNF context
        self.context_dim_cnf = self.summary_stats_dim + self.latent_dim_learned

        self.cnf_decoder = zuko.flows.NSF(
            features=2,
            context=self.context_dim_cnf,
            transforms=model_opts.get('zuko_nsf_num_flow_steps', 5),
            bins=model_opts.get('zuko_nsf_num_bins', 8),
            hidden_features=[model_opts.get('zuko_nsf_hidden_features', 128)] * \
                            model_opts.get('zuko_nsf_hidden_layers', 2),
            randperm=False
        )

    def _encode_transformer_to_latent_params(self, input_tq_sequence_norm_padded, attention_mask_src_key_padding):
        # input_tq_sequence_norm_padded: (batch, seq_len, 2)
        # attention_mask_src_key_padding: (batch, seq_len), True for valid, False for padding
        
        # 1. Embed (t,q)
        embedded_tq = self.input_tq_embedding(input_tq_sequence_norm_padded) # (batch, seq_len, d_model)
        
        # 2. Add positional encoding
        positioned_tq = self.positional_encoding(embedded_tq) # (batch, seq_len, d_model)
        
        # 3. Pass through Transformer encoder
        # src_key_padding_mask should be (N, S) where True means IGNORE.
        # Our attention_mask is True for VALID, so we need to invert it.
        src_key_padding_mask = ~attention_mask_src_key_padding if attention_mask_src_key_padding is not None else None

        transformer_output = self.transformer_encoder(
            positioned_tq,
            src_key_padding_mask=src_key_padding_mask
        ) # (batch, seq_len, d_model)
        
        # 4. Pool the output
        # Mean pooling over unpadded elements.
        # If src_key_padding_mask was used, those outputs might be zeroed or affected.
        # A robust way is to use the attention_mask_src_key_padding (original, True for valid)
        if attention_mask_src_key_padding is not None:
            mask_expanded = attention_mask_src_key_padding.unsqueeze(-1).expand_as(transformer_output)
            masked_output = transformer_output * mask_expanded
            sum_pooled_output = masked_output.sum(dim=1)
            num_valid_elements = attention_mask_src_key_padding.sum(dim=1, keepdim=True)
            # Avoid division by zero if all elements are padded (should not happen with valid inputs)
            pooled_output = sum_pooled_output / torch.clamp(num_valid_elements, min=1)
        else: # No padding, simple mean pool
            pooled_output = transformer_output.mean(dim=1) # (batch, d_model)
            
        # 5. Project pooled output to get mu_z_learned and log_sigma_sq_z_learned
        latent_params = self.latent_projection_head(pooled_output) # (batch, 2 * latent_dim_learned)
        mu_z_learned = latent_params[:, :self.latent_dim_learned]
        log_sigma_sq_z_learned = latent_params[:, self.latent_dim_learned:]
        
        return mu_z_learned, log_sigma_sq_z_learned

    def reparameterize(self, mu, log_sigma_sq):
        std = torch.exp(0.5 * log_sigma_sq)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --- PyTorch Lightning Methods ---
    def forward(self, batch):
        # For training/validation/test steps
        input_tq_sequence = batch["input_tq_sequence"]     # (N, S, 2), normalized, padded
        attention_mask = batch["attention_mask"]            # (N, S), True for valid
        z_summary_norm_batch = batch["z_summary"]           # (N, 10), normalized
        sensor_pos_norm_batch = batch["sensor_position"]    # (N, 3), normalized

        mu_z_learned, log_sigma_sq_z_learned = self._encode_transformer_to_latent_params(
            input_tq_sequence,
            attention_mask
        )
        z_learned_sampled = self.reparameterize(mu_z_learned, log_sigma_sq_z_learned) # (N, latent_dim_learned)
        
        # Apply FiLM using sensor_pos_norm_batch
        film_gamma, film_beta = self.film_layer(sensor_pos_norm_batch) # (N, latent_dim_learned), (N, latent_dim_learned)
        z_learned_modulated = film_gamma * z_learned_sampled + film_beta
        
        context_for_cnf = torch.cat([z_summary_norm_batch, z_learned_modulated], dim=1)
        
        return mu_z_learned, log_sigma_sq_z_learned, context_for_cnf

    def _shared_step(self, batch, batch_idx, stage_name):
        target_tq_pairs_for_loss = batch["input_tq_sequence"] # (N, S, 2), norm, padded
        # active_entries_count is the number of valid pulses/bins in target_tq_pairs_for_loss for each item
        # Shape (N), float tensor from dataloader.
        active_entries_count_for_masking = batch["active_entries_count"]

        mu_z_learned, log_sigma_sq_z_learned, context_for_cnf = self.forward(batch)

        # Expand context for sequence broadcasting: (batch_size, context_dim) -> (batch_size, 1, context_dim)
        context_for_cnf_expanded = context_for_cnf.unsqueeze(1)
        log_p_tq_given_context = self.cnf_decoder(context_for_cnf_expanded).log_prob(target_tq_pairs_for_loss)
        # log_p_tq_given_context shape: (batch_size, max_seq_len)

        # Create mask for reconstruction loss based on active_entries_count_for_masking
        # Mask should be (batch_size, max_seq_len)
        seq_indices = torch.arange(target_tq_pairs_for_loss.size(1), device=self.device)[None, :] # (1, max_seq_len)
        # active_entries_count_for_masking is (batch_size), needs to be (batch_size, 1) for broadcasting
        reconstruction_mask = seq_indices < active_entries_count_for_masking[:, None]
        
        # Apply mask and sum over sequence dimension
        masked_log_p = log_p_tq_given_context * reconstruction_mask
        reconstruction_loss_per_item = -torch.sum(masked_log_p, dim=1) # Sum over seq_len
        
        reconstruction_loss = reconstruction_loss_per_item.mean() # Average over batch

        # KL Divergence (for z_learned)
        kl_div = -0.5 * torch.sum(1 + log_sigma_sq_z_learned - mu_z_learned.pow(2) - log_sigma_sq_z_learned.exp(), dim=1)
        kl_div = kl_div.mean() # Mean over batch

        # Total ELBO Loss (negative ELBO to minimize) with beta annealing
        t_opts = self.hparams.training_options
        beta_final = t_opts.get('beta_kl', 1.0)
        beta_start = t_opts.get('beta_kl_start', 0.0)
        warmup = t_opts.get('beta_kl_warmup_epochs', 0)
        if warmup > 0:
            progress = min(self.current_epoch / warmup, 1.0)
            beta = beta_start + progress * (beta_final - beta_start)
        else:
            beta = beta_final
        loss = reconstruction_loss + beta * kl_div
        # Log current beta value
        self.log(f'{stage_name}_beta', beta, on_step=(stage_name=='train'), on_epoch=True, logger=True)

        self.log(f'{stage_name}_loss', loss, on_step=(stage_name=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=target_tq_pairs_for_loss.shape[0])
        self.log(f'{stage_name}_recon_loss', reconstruction_loss, on_step=(stage_name=='train'), on_epoch=True, logger=True, batch_size=target_tq_pairs_for_loss.shape[0])
        self.log(f'{stage_name}_kl_div', kl_div, on_step=(stage_name=='train'), on_epoch=True, logger=True, batch_size=target_tq_pairs_for_loss.shape[0])
        if stage_name == 'train':
            self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        opt_cfg = self.hparams.training_options
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.get('lr', 1e-4),
            weight_decay=opt_cfg.get('weight_decay', 0.01)
        )
        
        scheduler_params = opt_cfg.get('lr_schedule', None)
        if scheduler_params and isinstance(scheduler_params, list) and len(scheduler_params) == 2:
            T_max_epochs = self.hparams.training_options.get('epochs', 100)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_epochs, # Or calculate total steps: self.trainer.estimated_stepping_batches
                eta_min=1e-6
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] # "step" or "epoch"
        
        return optimizer
    
    # --- Public API Methods ---
    def encode(self, t_values_raw, q_values_raw, sensor_pos_raw, sample_z_learned=False):
        """ User-facing method to get latent representation from raw inputs for a single sensor or batch. """
        self.eval() # Ensure model is in eval mode for consistent behavior (e.g. dropout)
        
        # Ensure inputs are tensors and on the correct device
        if not isinstance(t_values_raw, torch.Tensor): t_values_raw = torch.tensor(t_values_raw, dtype=torch.float32)
        if not isinstance(q_values_raw, torch.Tensor): q_values_raw = torch.tensor(q_values_raw, dtype=torch.float32)
        if not isinstance(sensor_pos_raw, torch.Tensor): sensor_pos_raw = torch.tensor(sensor_pos_raw, dtype=torch.float32)

        t_values_raw = t_values_raw.to(self.device)
        q_values_raw = q_values_raw.to(self.device)
        sensor_pos_raw = sensor_pos_raw.to(self.device)

        is_batched = t_values_raw.ndim > 1
        if not is_batched: # If single instance, unsqueeze to make it a batch of 1
            t_values_raw = t_values_raw.unsqueeze(0)
            q_values_raw = q_values_raw.unsqueeze(0)
            sensor_pos_raw = sensor_pos_raw.unsqueeze(0)
        
        batch_size = t_values_raw.shape[0]
        input_tq_sequences_norm_padded_list = []
        attention_masks_list = []
        z_summaries_norm_list = []

        for i in range(batch_size):
            # Calculate z_summary (raw)
            # summary_stats_calculator expects numpy arrays
            current_t_np = t_values_raw[i].cpu().numpy()
            current_q_np = q_values_raw[i].cpu().numpy()
            z_summary_np_raw = self.summary_stats_calculator(current_t_np, current_q_np)
            z_summary_tensor_raw = torch.from_numpy(z_summary_np_raw).to(self.device)
            z_summary_norm = self.normalize_z_summary(z_summary_tensor_raw)
            z_summaries_norm_list.append(z_summary_norm)

            # Normalize t, q
            t_norm = self.normalize_tq(t_values_raw[i])
            q_norm = self.normalize_tq(q_values_raw[i])

            # Pad/truncate t_norm, q_norm
            current_seq_len = len(t_norm)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.bool, device=self.device)

            if current_seq_len > self.max_seq_len:
                # Simple truncation (take first self.max_seq_len)
                # Consider sorting by time first if not already sorted
                t_norm_padded = t_norm[:self.max_seq_len]
                q_norm_padded = q_norm[:self.max_seq_len]
            elif current_seq_len < self.max_seq_len:
                pad_len = self.max_seq_len - current_seq_len
                t_norm_padded = torch.cat([t_norm, torch.zeros(pad_len, device=self.device)], dim=0)
                q_norm_padded = torch.cat([q_norm, torch.zeros(pad_len, device=self.device)], dim=0)
                attention_mask[current_seq_len:] = False
            else:
                t_norm_padded = t_norm
                q_norm_padded = q_norm
            
            input_tq_sequence_norm_padded = torch.stack([t_norm_padded, q_norm_padded], dim=1)
            input_tq_sequences_norm_padded_list.append(input_tq_sequence_norm_padded)
            attention_masks_list.append(attention_mask)

        # Batch items for encoder
        batched_input_tq_norm_padded = torch.stack(input_tq_sequences_norm_padded_list, dim=0)
        batched_attention_masks = torch.stack(attention_masks_list, dim=0)
        batched_z_summaries_norm = torch.stack(z_summaries_norm_list, dim=0)
        
        # Normalize sensor_pos
        sensor_pos_norm = self.normalize_sensor_pos(sensor_pos_raw) # (batch, 3)

        # Encode
        mu_z_learned, log_sigma_sq_z_learned = self._encode_transformer_to_latent_params(
            batched_input_tq_norm_padded,
            batched_attention_masks
        )

        if sample_z_learned:
            z_learned_out = self.reparameterize(mu_z_learned, log_sigma_sq_z_learned)
        else:
            z_learned_out = mu_z_learned
            
        z_full_representation_norm = torch.cat([batched_z_summaries_norm, z_learned_out], dim=-1)

        if not is_batched: # If input was single, squeeze batch dim out
            z_full_representation_norm = z_full_representation_norm.squeeze(0)
            sensor_pos_norm = sensor_pos_norm.squeeze(0)
            mu_z_learned = mu_z_learned.squeeze(0)
            log_sigma_sq_z_learned = log_sigma_sq_z_learned.squeeze(0)

        return z_full_representation_norm, sensor_pos_norm, mu_z_learned, log_sigma_sq_z_learned

    def decode(self, z_full_representation_norm, sensor_pos_norm, N_to_generate=None):
        """ User-facing method to generate raw (t,q) data from normalized latent Z and sensor_pos. """
        self.eval()
        
        if not isinstance(z_full_representation_norm, torch.Tensor): z_full_representation_norm = torch.tensor(z_full_representation_norm, dtype=torch.float32)
        if not isinstance(sensor_pos_norm, torch.Tensor): sensor_pos_norm = torch.tensor(sensor_pos_norm, dtype=torch.float32)
        if not isinstance(N_to_generate, torch.Tensor): N_to_generate = torch.tensor(N_to_generate, dtype=torch.long)

        z_full_representation_norm = z_full_representation_norm.to(self.device)
        sensor_pos_norm = sensor_pos_norm.to(self.device)
        N_to_generate = N_to_generate.to(self.device)

        is_batched = z_full_representation_norm.ndim > 1
        if not is_batched:
            z_full_representation_norm = z_full_representation_norm.unsqueeze(0)
            sensor_pos_norm = sensor_pos_norm.unsqueeze(0)
            N_to_generate = N_to_generate.unsqueeze(0)

        batch_size = z_full_representation_norm.shape[0]
        
        # Apply FiLM to z_learned part
        # z_full_representation_norm is [z_summary_norm, z_learned_norm_unmodulated]
        z_summary_part = z_full_representation_norm[:, :self.summary_stats_dim]
        z_learned_unmodulated_part = z_full_representation_norm[:, self.summary_stats_dim:]
        
        film_gamma, film_beta = self.film_layer(sensor_pos_norm)
        z_learned_modulated_part = film_gamma * z_learned_unmodulated_part + film_beta
        
        context_for_decode = torch.cat([z_summary_part, z_learned_modulated_part], dim=-1)
        
        if N_to_generate is None:
            # If N_to_generate is not provided, use the first element of z_summary_part
            N_to_generate = z_summary_part[:, 0].long()
        max_n = N_to_generate.max().item() # Maximum number of samples to generate
        
        # Expand context for sequence broadcasting: (batch_size, context_dim) -> (batch_size, 1, context_dim)
        context_for_decode_expanded = context_for_decode.unsqueeze(1)
        samples_norm_permuted = self.cnf_decoder(context_for_decode_expanded).sample(torch.Size([max_n]))
        samples_norm = samples_norm_permuted.permute(1, 0, 2) # (batch_size, max_n, 2)

        t_samples_norm = samples_norm[..., 0] # (batch, max_n)
        q_samples_norm = samples_norm[..., 1] # (batch, max_n)

        t_samples_raw = self.denormalize_tq(t_samples_norm)
        q_samples_raw = self.denormalize_tq(q_samples_norm)
        
        # Create a list of (t,q) tensors, each truncated to its respective N_to_generate[i]
        output_tq_list = []
        for i in range(batch_size):
            n_gen = N_to_generate[i].item()
            if n_gen > 0:
                t_i = t_samples_raw[i, :n_gen]
                q_i = q_samples_raw[i, :n_gen]
                # Sort by time
                sort_indices = torch.argsort(t_i)
                t_i_sorted = t_i[sort_indices]
                q_i_sorted = q_i[sort_indices]
                output_tq_list.append(torch.stack([t_i_sorted, q_i_sorted], dim=1))
            else:
                output_tq_list.append(torch.empty((0,2), device=self.device, dtype=torch.float32))
        
        return output_tq_list if is_batched else output_tq_list[0]