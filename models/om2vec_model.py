import sys
import os
# Ensure the project root is in sys.path for direct script execution
# This allows imports like `from utils.om_processing import ...` to work
# when running `python3 models/om2vec_model.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Now this import should work when script is run directly
from utils.om_processing import calculate_summary_statistics, preprocess_photon_sequence, sinusoidal_positional_encoding
from models.flows import RealNVP1D # Import RealNVP1D

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Generates scale (gamma) and shift (beta) parameters from a context vector
    and applies them to an input.
    """
    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        # Network to generate gamma and beta from context
        # Output dimension is 2 * feature_dim (for gamma and beta separately)
        self.generator = nn.Linear(context_dim, 2 * feature_dim)
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.
        Args:
            features (torch.Tensor): Input features to modulate.
                                     Shape: (batch_size, ..., feature_dim)
            context (torch.Tensor): Context vector to generate FiLM parameters.
                                    Shape: (batch_size, context_dim)
        Returns:
            torch.Tensor: Modulated features. Shape: (batch_size, ..., feature_dim)
        """
        # Generate gamma and beta
        gamma_beta = self.generator(context) # (batch_size, 2 * feature_dim)
        
        # Split into gamma and beta
        gamma = gamma_beta[:, :self.feature_dim] # (batch_size, feature_dim)
        beta = gamma_beta[:, self.feature_dim:]  # (batch_size, feature_dim)

        # Reshape gamma and beta to be broadcastable with features
        # Features: (N_valid, P, D_emb)
        # Context: (N_valid, D_sensor_emb)
        # Gamma/Beta from generator: (N_valid, 2 * D_emb)
        # Gamma/Beta after split: (N_valid, D_emb)
        # We need to unsqueeze to make them (N_valid, 1, D_emb) to broadcast over P
        gamma = gamma.unsqueeze(1) # (N_valid, 1, D_emb)
        beta = beta.unsqueeze(1)   # (N_valid, 1, D_emb)

        return gamma * features + beta

class Om2vecModel(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg) # Saves cfg to self.hparams
        self.cfg = cfg # Also keep it directly accessible
        self.model_cfg = cfg['model_options']
        self.training_cfg = cfg['training_options']
        self.data_cfg = cfg['data_options']

        # --- Encoder Submodules ---
        # 1. Input Embedding Layer for (t,q) sequences
        # Input: (batch_size * num_oms, max_photons_per_om, 2) for (t,q)
        # Output: (batch_size * num_oms, max_photons_per_om, input_embedding_dim)
        # Note: Actual (t,q) values are raw; normalization happens in preprocess_photon_sequence
        # or just before this layer. For now, assuming (t,q) are passed to preprocess_photon_sequence
        # and then the output of that is embedded.
        # The input to this layer will be the (normalized_t, normalized_q) pairs.
        self.input_embedder = nn.Linear(2, self.model_cfg['input_embedding_dim'])

        # 2. Positional Encoding (will be added to embedded input)
        # Max length for positional encoding is max_photons_per_om
        self.pos_encoder = sinusoidal_positional_encoding(
            max_len=self.data_cfg['max_photons_per_om'],
            d_model=self.model_cfg['input_embedding_dim']
        )
        # self.register_buffer('pos_encoder', self.pos_encoder_tensor, persistent=False)


        # 3. Optional Sensor Position Embedding MLP
        self.sensor_pos_embedder = None
        if self.model_cfg.get('sensor_pos_embedding_dim') and self.model_cfg.get('sensor_integration_type') != 'none':
            self.sensor_pos_embedder = nn.Linear(3, self.model_cfg['sensor_pos_embedding_dim'])
            if self.model_cfg.get('sensor_integration_type') == 'film':
                self.film_layer = FiLMLayer(
                    context_dim=self.model_cfg['sensor_pos_embedding_dim'],
                    feature_dim=self.model_cfg['input_embedding_dim']
                )

        # 4. Transformer Encoder
        # Determine the actual input feature dimension for the Transformer
        # FiLM is applied to the input embeddings, so transformer d_model remains input_embedding_dim
        transformer_feature_dim = self.model_cfg['input_embedding_dim']
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_feature_dim,
            nhead=self.model_cfg['transformer_num_heads'],
            dim_feedforward=self.model_cfg.get('transformer_feedforward_dim', transformer_feature_dim * 4), # Often 4*d_model
            dropout=self.model_cfg['transformer_dropout'],
            activation=self.model_cfg.get('transformer_activation', 'gelu'), # 'relu' or 'gelu'
            batch_first=True,
            norm_first=self.model_cfg.get('transformer_norm_first', False)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.model_cfg['transformer_num_layers']
        )

        # 5. Pooling Layer (after Transformer)
        # Strategy: 'mean' or 'cls'. Implemented in forward.

        # 6. Latent Variable Projection Layers (for z_learned)
        # Input dimension is transformer_feature_dim (output dim of Transformer)
        self.fc_mu_learned = nn.Linear(transformer_feature_dim, self.model_cfg['latent_learned_dim'])
        self.fc_log_sigma_sq_learned = nn.Linear(transformer_feature_dim, self.model_cfg['latent_learned_dim'])

        # --- Decoder Submodules (CNF - Conditional Normalizing Flow) ---
        self.latent_summary_dim = 9 # Fixed number of summary statistics
        
        # Determine the context dimension for the CNF hypernetwork
        # Context = z_summary (9) + z_learned (latent_learned_dim)
        self.cnf_context_dim = self.latent_summary_dim + self.model_cfg['latent_learned_dim']
        
        # self.cnf_conditions_on_sensor_pos is no longer used as CNF does not condition on sensor position.
        # The cnf_context_dim is solely based on z_summary and z_learned.

        # Instantiate RealNVP1D
        self.cnf_decoder = RealNVP1D(
            input_dim=self.model_cfg.get('cnf_base_dist_dim', 1),
            context_dim=self.cnf_context_dim,
            num_coupling_layers=self.model_cfg.get('cnf_num_layers', 5), # cnf_num_layers maps to num_coupling_layers
            hidden_dims_s_t_net=self.model_cfg.get('cnf_hidden_dims_hypernet', [128, 128]),
            activation_s_t_net=self.model_cfg.get('cnf_activation_s_t_net', 'relu') # Allow configuring activation
        )

        # KL Annealing factor
        self.current_kl_beta = self.training_cfg.get('kl_beta_initial_value', 0.0)


    def forward(self, batch: dict):
        """
        Processes a batch of event data for training/validation.
        Uses the encode method to get latent variables and then calculates
        reconstruction loss using the CNF decoder.

        Args:
            batch (dict): A dictionary of batched tensors from the dataloader.
                          Expected keys: "all_om_hits", "all_om_sensor_pos",
                                         "om_mask", "hit_mask".
        Returns:
            dict: Containing "reconstruction_log_prob", "kl_divergence",
                  "num_valid_oms", and other intermediate tensors for logging/debugging.
        """
        batch_size, max_oms_in_event, max_photons_per_om, _ = batch['all_om_hits'].shape
        device = batch['all_om_hits'].device

        # --- Identify and gather data for valid OMs ---
        om_mask_flat = batch['om_mask'].view(-1)
        valid_om_indices_flat = torch.where(om_mask_flat)[0]

        if valid_om_indices_flat.numel() == 0:
            return {
                "reconstruction_log_prob": torch.tensor(0.0, device=device),
                "kl_divergence": torch.tensor(0.0, device=device),
                "num_valid_oms": 0
            }

        all_om_hits_flat = batch['all_om_hits'].view(-1, max_photons_per_om, 2)
        raw_tq_for_valid_oms = all_om_hits_flat[valid_om_indices_flat]

        all_om_sensor_pos_flat = batch['all_om_sensor_pos'].view(-1, 3)
        sensor_pos_for_valid_oms = all_om_sensor_pos_flat[valid_om_indices_flat]
        
        hit_mask_flat = batch['hit_mask'].view(-1, max_photons_per_om)
        hit_mask_for_valid_oms = hit_mask_flat[valid_om_indices_flat]

        # --- Encode: Get z_summary, mu_learned, log_sigma_sq_learned ---
        # Pass raw_tq_for_valid_oms, hit_mask_for_valid_oms, sensor_pos_for_valid_oms
        z_summary_stack, mu_learned, log_sigma_sq_learned = self.encode(
            raw_tq_sequences=raw_tq_for_valid_oms,
            hit_masks=hit_mask_for_valid_oms,
            sensor_positions=sensor_pos_for_valid_oms,
            return_dist_params=True
        ) # z_summary (N_valid,9), mu (N_valid,D_l), log_sigma_sq (N_valid,D_l)
        
        # --- Sample z_learned (Reparameterization Trick) ---
        std_learned = torch.exp(0.5 * log_sigma_sq_learned)
        epsilon = torch.randn_like(std_learned, device=device)
        z_learned_sampled = mu_learned + std_learned * epsilon # (N_valid, D_learned)

        # --- Calculate KL Divergence for z_learned for each OM ---
        kl_div_per_om = -0.5 * torch.sum(
            1 + log_sigma_sq_learned - mu_learned.pow(2) - log_sigma_sq_learned.exp(),
            dim=1
        ) # (N_valid)
        total_kl_divergence = torch.sum(kl_div_per_om)

        # --- CNF Decoder for Reconstruction Log Probability ---
        # 1. Form full latent vector z_full for CNF context
        z_full_for_cnf_context = torch.cat((z_summary_stack, z_learned_sampled), dim=1)

        # 2. Prepare actual context for CNF (potentially including sensor pos)
        # CNF no longer conditions on sensor position, so cnf_actual_context is just z_full.
        cnf_actual_context = z_full_for_cnf_context
        
        # 3. Gather observed photon times (targets for CNF) and their corresponding contexts
        all_valid_photon_times_list = []
        all_valid_photon_contexts_list = []
        
        for i in range(valid_om_indices_flat.numel()): # Loop N_valid OMs
            om_raw_times = raw_tq_for_valid_oms[i, :, 0]
            om_hit_mask = hit_mask_for_valid_oms[i]
            actual_om_times = om_raw_times[om_hit_mask] # (N_hits_this_om,)
            
            if actual_om_times.numel() > 0:
                all_valid_photon_times_list.append(actual_om_times.unsqueeze(-1)) # (N_hits_this_om, 1)
                # Repeat the specific context for this OM for all its photons
                om_context_repeated = cnf_actual_context[i].unsqueeze(0).expand(actual_om_times.numel(), -1)
                all_valid_photon_contexts_list.append(om_context_repeated)

        if not all_valid_photon_times_list:
            total_reconstruction_log_prob = torch.tensor(0.0, device=device)
        else:
            all_valid_photon_times_cat = torch.cat(all_valid_photon_times_list, dim=0) # Shape: (Total_valid_photons, 1)
            all_valid_photon_contexts_cat = torch.cat(all_valid_photon_contexts_list, dim=0) # Shape: (Total_valid_photons, context_dim)
            
            # Normalize target times for CNF: t_norm = log(t_raw + offset)
            tq_log_norm_offset = self.data_cfg.get('tq_log_norm_offset', 1.0)
            norm_target_times_cat = torch.log(all_valid_photon_times_cat.clamp(min=0) + tq_log_norm_offset)

            log_probs_norm_times = self.cnf_decoder.log_prob(
                inputs=norm_target_times_cat, # CNF sees normalized times
                context=all_valid_photon_contexts_cat
            ) # Shape: (Total_valid_photons,)

            # Jacobian correction for the transformation t_norm = log(t_raw + offset)
            # log P(t_raw|z) = log P(t_norm|z) - log(t_raw + offset)
            # The subtraction term is -log(t_raw + offset), which is -norm_target_times_cat (if offset is same as used for norm_target_times_cat)
            # No, it's -log(t_raw + offset_for_jacobian_correction).
            # The derivative dt_norm/dt_raw = 1 / (t_raw + offset). So log_abs_det_jacobian = -log(t_raw + offset).
            # We must use the same offset for this correction as used in the transformation.
            log_abs_det_jacobian = -torch.log(all_valid_photon_times_cat.clamp(min=0) + tq_log_norm_offset).squeeze(-1) # Squeeze from (N,1) to (N,)

            # Corrected log probability for raw times
            log_probs_photons_corrected = log_probs_norm_times + log_abs_det_jacobian
            
            total_reconstruction_log_prob = torch.sum(log_probs_photons_corrected)

        return {
            "reconstruction_log_prob": total_reconstruction_log_prob,
            "kl_divergence": total_kl_divergence,
            "num_valid_oms": valid_om_indices_flat.numel(),
            "mu_learned": mu_learned,
            "log_sigma_sq_learned": log_sigma_sq_learned,
            "z_summary_stack": z_summary_stack,
            "z_learned_sampled": z_learned_sampled
        }

    def encode(self, raw_tq_sequences: torch.Tensor,
               hit_masks: torch.Tensor,
               sensor_positions: torch.Tensor,
               return_dist_params: bool = False):
        """
        Encodes a batch of OM photon sequences into their latent representations.

        Args:
            raw_tq_sequences (torch.Tensor): Raw (time, charge) photon sequences.
                                             Shape: (N_oms, max_photons_per_om, 2)
            hit_masks (torch.Tensor): Boolean mask for valid photons in sequences.
                                      Shape: (N_oms, max_photons_per_om)
            sensor_positions (torch.Tensor): Sensor (x,y,z) positions.
                                             Shape: (N_oms, 3)
            return_dist_params (bool): If True, returns mu and log_sigma_sq for z_learned.
                                       Otherwise, returns sampled z_learned (if training) or mu_learned (if eval).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                z_summary (N_oms, 9),
                z_learned_representation (N_oms, latent_learned_dim)
                (this is mu_learned if not return_dist_params and self.evaluating,
                 or sampled z_learned if not return_dist_params and self.training,
                 or (mu_learned, log_sigma_sq_learned) if return_dist_params)
        """
        N_oms, max_photons_per_om, _ = raw_tq_sequences.shape
        device = raw_tq_sequences.device

        # 1. Calculate Summary Statistics (z_summary)
        z_summaries_list = []
        for i in range(N_oms):
            om_hits_padded = raw_tq_sequences[i]
            om_hit_mask = hit_masks[i]
            actual_times = om_hits_padded[om_hit_mask, 0]
            actual_charges = om_hits_padded[om_hit_mask, 1]
            
            if actual_times.numel() == 0:
                default_stat_val = torch.log(torch.tensor(self.data_cfg.get('time_log_epsilon', 1e-9), device=device))
                z_summary_om = torch.full((self.latent_summary_dim,), default_stat_val, device=device, dtype=torch.float32)
            else:
                # Assuming summary statistics also adopt log(x+1) normalization for consistency
                # The calculate_summary_statistics function uses its arguments for this.
                # We'll pass the tq_log_norm_offset for both.
                current_tq_log_norm_offset = self.data_cfg.get('tq_log_norm_offset', 1.0)
                z_summary_om = calculate_summary_statistics(
                    actual_times, actual_charges,
                    charge_log_offset=current_tq_log_norm_offset,
                    time_log_epsilon=current_tq_log_norm_offset # Using the same offset for time's "epsilon"
                )
            z_summaries_list.append(z_summary_om)
        z_summary_stack = torch.stack(z_summaries_list) # (N_oms, 9)

        # 2. Preprocess (t,q) sequences for Transformer input (Normalization)
        tq_log_norm_offset = self.data_cfg.get('tq_log_norm_offset', 1.0)
        valid_om_times_padded = raw_tq_sequences[..., 0]
        valid_om_charges_padded = raw_tq_sequences[..., 1]
        norm_times = torch.log(valid_om_times_padded.clamp(min=0) + tq_log_norm_offset)
        norm_charges = torch.log(valid_om_charges_padded.clamp(min=0) + tq_log_norm_offset)
        normalized_tq_sequences = torch.stack((norm_times, norm_charges), dim=-1) # (N_oms, P, 2)

        # 3. Embed (t,q) sequence + Positional Encoding
        embedded_tq = self.input_embedder(normalized_tq_sequences) # (N_oms, P, D_emb)
        embedded_tq += self.pos_encoder.to(embedded_tq.device)[:max_photons_per_om, :]

        # 4. Optional Sensor Position Integration (FiLM)
        transformer_input = embedded_tq
        if self.model_cfg.get('sensor_integration_type') == 'film' and \
           self.sensor_pos_embedder is not None and \
           hasattr(self, 'film_layer') and \
           self.model_cfg.get('sensor_pos_embedding_dim', 0) > 0:
            sensor_pos_norm_scale = self.data_cfg.get('sensor_pos_norm_scale', 1.0)
            norm_sensor_pos = sensor_positions / sensor_pos_norm_scale
            sensor_pos_emb = self.sensor_pos_embedder(norm_sensor_pos)
            transformer_input = self.film_layer(transformer_input, sensor_pos_emb)
        
        # 5. Pass through Transformer Encoder
        transformer_padding_mask = ~hit_masks
        transformer_output = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=transformer_padding_mask
        ) # (N_oms, P, D_emb)

        # 6. Pool Transformer Outputs
        if self.model_cfg.get('pooling_strategy', 'mean') == 'mean':
            expanded_hit_mask = hit_masks.unsqueeze(-1).expand_as(transformer_output)
            summed_output = (transformer_output * expanded_hit_mask).sum(dim=1)
            num_actual_hits = hit_masks.sum(dim=1).unsqueeze(-1).clamp(min=1)
            pooled_output = summed_output / num_actual_hits
        elif self.model_cfg.get('pooling_strategy') == 'cls':
            raise NotImplementedError("CLS token pooling not yet implemented for encode method.")
        else:
            raise ValueError(f"Unknown pooling strategy: {self.model_cfg.get('pooling_strategy')}")

        # 7. Get mu_learned, log_sigma_sq_learned for z_learned
        mu_learned = self.fc_mu_learned(pooled_output) # (N_oms, D_learned)
        log_sigma_sq_learned = self.fc_log_sigma_sq_learned(pooled_output) # (N_oms, D_learned)

        if return_dist_params:
            return z_summary_stack, mu_learned, log_sigma_sq_learned
        
        # If not returning distribution parameters, return a single representation for z_learned
        if self.training: # Use sampled z_learned during training if not returning params
            std_learned = torch.exp(0.5 * log_sigma_sq_learned)
            epsilon = torch.randn_like(std_learned, device=device)
            z_learned_representation = mu_learned + std_learned * epsilon
        else: # Use mu_learned during evaluation
            z_learned_representation = mu_learned
            
        return z_summary_stack, z_learned_representation

    def decode(self, z_summary: torch.Tensor,
               z_learned: torch.Tensor,
               # sensor_positions is removed as CNF no longer conditions on it.
               times_to_evaluate: torch.Tensor = None,
               num_time_bins: int = None,
               time_range: tuple = None):
        """
        Decodes latent representations (z_summary, z_learned) to produce photon arrival time PDFs.

        Args:
            z_summary (torch.Tensor): Summary statistics part of the latent vector. Shape: (N_oms, 9)
            z_learned (torch.Tensor): Learned part of the latent vector. Shape: (N_oms, latent_learned_dim)
            times_to_evaluate (torch.Tensor, optional): Specific time points to evaluate the PDF at.
                                                        Shape: (N_oms, num_eval_times) or (num_eval_times,).
                                                        If provided, num_time_bins and time_range are ignored.
            num_time_bins (int, optional): Number of time bins to generate for the PDF. Used if times_to_evaluate is None.
            time_range (tuple, optional): (min_time, max_time) for generating time bins. Used if times_to_evaluate is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                eval_times (N_oms, num_eval_points): The time points at which the PDF was evaluated.
                log_pdf_values (N_oms, num_eval_points): The log P(t|z) values at those time points.
                                                         To get P(t|z), take torch.exp().
        """
        N_oms = z_summary.shape[0]
        device = z_summary.device

        # 1. Form full latent vector z_full
        z_full = torch.cat((z_summary, z_learned), dim=1) # (N_oms, 9 + D_learned)

        # 2. Prepare context for CNF
        # CNF no longer conditions on sensor position, so cnf_context is just z_full.
        cnf_context = z_full

        # 3. Determine time points for PDF evaluation
        if times_to_evaluate is None:
            if num_time_bins is None or time_range is None:
                raise ValueError("Either times_to_evaluate or (num_time_bins and time_range) must be provided.")
            # Generate time points for each OM in the batch
            # eval_times_single_om = torch.linspace(time_range[0], time_range[1], num_time_bins, device=device)
            # eval_times = eval_times_single_om.unsqueeze(0).expand(N_oms, -1) # (N_oms, num_time_bins)
            # For nflows, input needs to be (total_samples, 1) and context (total_samples, context_dim)
            # So we create a large batch of times and contexts
            _eval_times_list = []
            _contexts_list = []
            for i in range(N_oms):
                eval_times_single_om = torch.linspace(time_range[0], time_range[1], num_time_bins, device=device)
                _eval_times_list.append(eval_times_single_om.unsqueeze(-1)) # (num_time_bins, 1)
                _contexts_list.append(cnf_context[i].unsqueeze(0).expand(num_time_bins, -1)) # (num_time_bins, context_dim)
            
            eval_times_flat = torch.cat(_eval_times_list, dim=0) # (N_oms * num_time_bins, 1)
            contexts_flat = torch.cat(_contexts_list, dim=0) # (N_oms * num_time_bins, context_dim)
            output_eval_times_shape = (N_oms, num_time_bins)

        else: # times_to_evaluate is provided
            if times_to_evaluate.ndim == 1: # (num_eval_times,)
                # Expand for each OM in the batch and then flatten
                _eval_times_list = []
                _contexts_list = []
                for i in range(N_oms):
                    _eval_times_list.append(times_to_evaluate.unsqueeze(-1)) # (num_eval_times, 1)
                    _contexts_list.append(cnf_context[i].unsqueeze(0).expand(times_to_evaluate.shape[0], -1))
                eval_times_flat = torch.cat(_eval_times_list, dim=0)
                contexts_flat = torch.cat(_contexts_list, dim=0)
                output_eval_times_shape = (N_oms, times_to_evaluate.shape[0])

            elif times_to_evaluate.ndim == 2 and times_to_evaluate.shape[0] == N_oms: # (N_oms, num_eval_times)
                # Flatten for nflows
                _eval_times_list = []
                _contexts_list = []
                for i in range(N_oms):
                    _eval_times_list.append(times_to_evaluate[i].unsqueeze(-1)) # (num_eval_times_i, 1)
                    _contexts_list.append(cnf_context[i].unsqueeze(0).expand(times_to_evaluate[i].shape[0], -1))
                eval_times_flat = torch.cat(_eval_times_list, dim=0)
                contexts_flat = torch.cat(_contexts_list, dim=0)
                output_eval_times_shape = times_to_evaluate.shape
            else:
                raise ValueError("times_to_evaluate has incorrect shape.")

        # 4. Calculate log P(t | context) using CNF
        # eval_times_flat contains raw times. Normalize them before passing to CNF.
        tq_log_norm_offset = self.data_cfg.get('tq_log_norm_offset', 1.0)
        norm_eval_times_flat = torch.log(eval_times_flat.clamp(min=0) + tq_log_norm_offset)

        log_pdf_norm_times_flat = self.cnf_decoder.log_prob(
            inputs=norm_eval_times_flat, # CNF sees normalized times
            context=contexts_flat
        ) # (N_oms * num_eval_points,)

        # Jacobian correction
        log_abs_det_jacobian_decode = -torch.log(eval_times_flat.clamp(min=0) + tq_log_norm_offset).squeeze(-1) # Squeeze from (N,1) to (N,)
        
        log_pdf_values_corrected_flat = log_pdf_norm_times_flat + log_abs_det_jacobian_decode
        
        log_pdf_values = log_pdf_values_corrected_flat.view(output_eval_times_shape) # Reshape to (N_oms, num_eval_points)
        
        # Prepare original eval_times in the (N_oms, num_eval_points) shape for returning
        # This part remains the same as it's about the shape of the `eval_times` being returned, not their values fed to CNF.
        if times_to_evaluate is None:
            eval_times_per_om = torch.linspace(time_range[0], time_range[1], num_time_bins, device=device)
            final_eval_times = eval_times_per_om.unsqueeze(0).expand(N_oms, -1)
        elif times_to_evaluate.ndim == 1:
            final_eval_times = times_to_evaluate.unsqueeze(0).expand(N_oms, -1)
        else: # times_to_evaluate.ndim == 2
            final_eval_times = times_to_evaluate
            
        return final_eval_times, log_pdf_values

    def _shared_step(self, batch: dict, batch_idx: int, stage: str):
        # `forward` now returns the necessary components for loss calculation
        outputs = self.forward(batch)
        
        num_valid_oms = outputs["num_valid_oms"]
        
        if num_valid_oms == 0:
            self.log(f'{stage}_num_valid_oms', float(num_valid_oms), batch_size=batch['all_om_hits'].shape[0])
            # Return None or a zero loss tensor if appropriate for Lightning
            # For now, returning None, assuming Lightning handles cases where loss is not computable.
            return None

        # ELBO = E[log P(X|z)] - beta * KL(q(z|X) || p(z))
        # We want to MAXIMIZE ELBO, so MINIMIZE -ELBO
        # Loss = - (sum_reconstruction_log_prob / num_valid_oms) + beta * (sum_kl_divergence / num_valid_oms)
        
        # reconstruction_log_prob and kl_divergence are already sums over all valid OMs in the batch.
        # We average them by the number of valid OMs.
        avg_reconstruction_log_prob = outputs["reconstruction_log_prob"] / num_valid_oms
        avg_kl_divergence = outputs["kl_divergence"] / num_valid_oms
        
        loss = -avg_reconstruction_log_prob + self.current_kl_beta * avg_kl_divergence

        # Logging
        batch_size_events = batch['all_om_hits'].shape[0] # Number of events in the batch
        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size_events)
        self.log(f'{stage}_recon_log_prob_avg_per_om', avg_reconstruction_log_prob, on_step=(stage=='train'), on_epoch=True, logger=True, batch_size=batch_size_events)
        self.log(f'{stage}_kl_div_avg_per_om', avg_kl_divergence, on_step=(stage=='train'), on_epoch=True, logger=True, batch_size=batch_size_events)
        self.log(f'{stage}_num_valid_oms_total', float(num_valid_oms), on_step=False, on_epoch=True, logger=True, batch_size=batch_size_events) # Total valid OMs in batch
        
        # Log average number of valid OMs per event in the batch
        avg_valid_oms_per_event = float(num_valid_oms) / batch_size_events if batch_size_events > 0 else 0
        self.log(f'{stage}_num_valid_oms_avg_per_event', avg_valid_oms_per_event, on_step=(stage=='train'), on_epoch=True, logger=True, batch_size=batch_size_events)

        if stage == 'train':
           self.log('kl_beta', self.current_kl_beta, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch: dict, batch_idx: int):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch: dict, batch_idx: int):
        return self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.training_cfg.get('lr', 1e-4),
            weight_decay=self.training_cfg.get('weight_decay', 0.01)
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_cfg.get('lr_schedule_t_max', self.trainer.max_epochs if self.trainer else 500),
            eta_min=self.training_cfg.get('lr_schedule_eta_min', 1e-6)
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # KL Annealing update
        if self.training_cfg.get('kl_anneal_epochs', 0) > 0 and self.current_epoch < self.training_cfg['kl_anneal_epochs']:
            initial_beta = self.training_cfg.get('kl_beta_initial_value', 0.0)
            final_beta = self.training_cfg.get('kl_beta_final_value', 1.0)
            anneal_epochs = self.training_cfg['kl_anneal_epochs']
            
            if self.training_cfg.get('kl_anneal_schedule', 'linear') == 'linear':
                self.current_kl_beta = initial_beta + (final_beta - initial_beta) * (self.current_epoch / anneal_epochs)
            # Add other schedules like cosine if needed
            self.current_kl_beta = min(self.current_kl_beta, final_beta) # Cap at final_beta
        elif self.current_epoch >= self.training_cfg.get('kl_anneal_epochs', 0):
            self.current_kl_beta = self.training_cfg.get('kl_beta_final_value', 1.0)
        
        # For the first epoch if no annealing
        if self.training_cfg.get('kl_anneal_epochs', 0) == 0:
             self.current_kl_beta = self.training_cfg.get('kl_beta_final_value', 1.0)