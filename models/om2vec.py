import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout_rate):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Using GRU as specified for simplicity in the plan
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Input and output tensors are provided as (batch, seq, feature)
            dropout=dropout_rate if num_layers > 1 else 0 # Dropout only if num_layers > 1
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, lengths):
        """
        Args:
            x (Tensor): Input sequence, shape (batch_size, seq_len, input_dim)
            lengths (Tensor): Original sequence lengths for packing, shape (batch_size,)
        
        Returns:
            mu (Tensor): Mean of the latent distribution, shape (batch_size, latent_dim)
            log_var (Tensor): Log variance of the latent distribution, shape (batch_size, latent_dim)
        """
        # Pack padded sequence
        # Ensure lengths are on CPU for pack_padded_sequence if they are not already
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, hidden = self.rnn(packed_x)
        
        # Output from GRU is (seq_len, batch, hidden_size * num_directions)
        # Hidden state from GRU is (num_layers * num_directions, batch, hidden_size)
        # We want the hidden state of the last layer
        # If batch_first=True for RNN, hidden is (D*num_layers, N, H_out)
        # We take the hidden state from the last layer.
        # If num_layers > 1, hidden will be (num_layers, batch_size, hidden_dim). We take hidden[-1]
        # If num_layers = 1, hidden will be (1, batch_size, hidden_dim). We take hidden[0]
        
        # The 'hidden' variable returned by GRU is the final hidden state for each element in the batch.
        # Its shape is (num_layers * num_directions, batch, hidden_size).
        # We need the hidden state of the last layer, last time step.
        # For a GRU, hidden[-1] gives the hidden state of the last layer for all sequences.
        last_layer_hidden = hidden[-1] # Shape: (batch_size, hidden_dim)
        
        mu = self.fc_mu(last_layer_hidden)
        log_var = self.fc_log_var(last_layer_hidden)
        
        return mu, log_var

class NTPPDecoder(nn.Module):
    def __init__(self, latent_dim, event_input_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.latent_dim = latent_dim
        self.event_input_dim = event_input_dim # For (log_inter_event_time, log_charge)
        self.hidden_dim = hidden_dim
        
        # Layer to project latent_dim (from VAE's z) to rnn_hidden_dim for initial hidden state
        self.fc_init_hidden = nn.Linear(latent_dim, num_layers * hidden_dim) # num_layers for GRU

        self.rnn = nn.GRU(
            input_size=event_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layers for parameters of inter-event time distribution (Exponential)
        # Predicting rate (lambda > 0), so use softplus
        self.fc_time_param = nn.Linear(hidden_dim, 1)
        
        # Output layers for parameters of charge distribution (Gaussian)
        self.fc_charge_mean = nn.Linear(hidden_dim, 1)
        self.fc_charge_log_std = nn.Linear(hidden_dim, 1)

    def forward(self, z, target_event_sequences, initial_hidden_state_override=None):
        """
        Autoregressive decoder for training (teacher forcing).
        
        Args:
            z (Tensor): Sampled latent variable, shape (batch_size, latent_dim). Used to initialize hidden state.
            target_event_sequences (Tensor): Ground truth event sequences for teacher forcing.
                                             Shape (batch_size, seq_len, event_input_dim).
                                             event_input_dim corresponds to (log_inter_event_time, log_charge).
            initial_hidden_state_override (Tensor, optional): For multi-step generation during inference.
                                                              Shape (num_layers, batch_size, hidden_dim).
        Returns:
            dict: Contains tensors of predicted distribution parameters for each time step.
                  'time_params': (batch_size, seq_len, 1) -> rate for Exponential
                  'charge_means': (batch_size, seq_len, 1) -> mean for Gaussian
                  'charge_log_stds': (batch_size, seq_len, 1) -> log_std for Gaussian
        """
        batch_size = target_event_sequences.size(0)
        seq_len = target_event_sequences.size(1)

        if initial_hidden_state_override is not None:
            h_0 = initial_hidden_state_override
        else:
            # Initialize hidden state from z
            # fc_init_hidden outputs (batch_size, num_layers * hidden_dim)
            # Reshape to (batch_size, num_layers, hidden_dim) then permute to (num_layers, batch_size, hidden_dim)
            init_hidden_flat = self.fc_init_hidden(z)
            h_0 = init_hidden_flat.view(batch_size, self.rnn.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # For teacher forcing, the input to RNN at step t is the target event from step t-1.
        # The first input can be a zero vector or a learned start-of-sequence token.
        # Here, target_event_sequences already includes the "previous" event features.
        # The NTPP model predicts (t_i, m_i) based on h_i (derived from (t_{i-1}, m_{i-1}) and h_{i-1}).
        # So, the input to the RNN at each step is the actual previous (log_dt, log_charge).
        
        rnn_outputs, _ = self.rnn(target_event_sequences, h_0)
        # rnn_outputs shape: (batch_size, seq_len, hidden_dim)
        
        time_params = F.softplus(self.fc_time_param(rnn_outputs)) # Ensure rate > 0
        charge_means = self.fc_charge_mean(rnn_outputs)
        charge_log_stds = self.fc_charge_log_std(rnn_outputs)
        
        return {
            "time_params": time_params,          # (batch_size, seq_len, 1)
            "charge_means": charge_means,        # (batch_size, seq_len, 1)
            "charge_log_stds": charge_log_stds   # (batch_size, seq_len, 1)
        }

class Om2vec(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # self.save_hyperparameters(cfg) # Saves the whole cfg, which can be large.
        # Instead, save specific model and training options.
        # Ensure all required keys are present in cfg['model_options'] and cfg['training_options']
        # For example, 'event_input_dim' is used below but was added to om2vec.cfg in a later step.
        # It should be present in the cfg passed to this constructor.
        self.save_hyperparameters(cfg['model_options'])
        self.save_hyperparameters(cfg['training_options'])
        
        self.data_cfg = cfg['data_options']
        self.model_cfg = cfg['model_options']

        self.encoder = VAEEncoder(
            input_dim=2, # (log_time_abs_grouped, log_count_grouped)
            hidden_dim=self.model_cfg['rnn_hidden_dim'],
            latent_dim=self.model_cfg['latent_dim'],
            num_layers=self.model_cfg['rnn_num_layers'],
            dropout_rate=self.model_cfg['dropout_rate']
        )
        
        self.decoder = NTPPDecoder(
            latent_dim=self.model_cfg['latent_dim'],
            event_input_dim=self.model_cfg.get('event_input_dim', 2), # Default to 2 if not in cfg
            hidden_dim=self.model_cfg['rnn_hidden_dim'],
            num_layers=self.model_cfg['rnn_num_layers'],
            dropout_rate=self.model_cfg['dropout_rate']
        )
        
        # It's generally better to import at the top of the file.
        # However, to keep this module self-contained for now or if utils might not always be available:
        from utils.utils import log_transform, nll_exponential, nll_gaussian, kl_divergence_gaussian
        self.log_transform_fn = log_transform # Renamed to avoid conflict if self.log_transform is a PL attribute
        self.nll_exponential = nll_exponential
        self.nll_gaussian = nll_gaussian
        self.kl_divergence_gaussian = kl_divergence_gaussian
        
        self.log_eps = self.data_cfg.get('log_transform_eps', 1e-6)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        # Input from dataloader:
        # batch["times_padded"] -> log-transformed absolute times of grouped events (B, L_pad)
        # batch["counts_padded"] -> log-transformed counts of grouped events (B, L_pad)
        # batch["raw_times_padded"] -> Non-transformed absolute times (B, L_pad)
        # batch["raw_counts_padded"] -> Non-transformed counts (B, L_pad)
        # batch["attention_mask"] -> Boolean mask (B, L_pad)
        # batch["sequence_lengths"] -> Original sequence lengths (B,)

        # 1. VAE Encoder
        encoder_input_seq = torch.stack([batch["times_padded"], batch["counts_padded"]], dim=2)
        mu, log_var = self.encoder(encoder_input_seq, batch["sequence_lengths"])
        z = self.reparameterize(mu, log_var)
        
        # 2. Prepare inputs and targets for NTPP Decoder (teacher forcing)
        
        # Target inter-event times (raw, for loss calculation)
        raw_times = batch["raw_times_padded"]
        first_event_placeholder = torch.zeros_like(raw_times[:, :1])
        times_for_diff = torch.cat([first_event_placeholder, raw_times], dim=1)
        target_delta_t_raw = torch.diff(times_for_diff, dim=1)[:, :raw_times.size(1)] # Ensure same length as original seq
        target_delta_t_raw = target_delta_t_raw * batch["attention_mask"].float()

        # Target counts/marks (raw, for loss calculation)
        target_counts_raw = batch["raw_counts_padded"] * batch["attention_mask"].float() # Already (B, L_pad)

        # Decoder RNN input: log-transformed (delta_t, count) of the *previous* event.
        # For the first event, delta_t is t_1 (time of first event), count is m_1.
        # The NTPPDecoder's RNN input at step j should be (log_delta_t_j, log_mark_j)
        # to predict parameters for (delta_t_{j+1}, mark_{j+1}).
        
        # Log-transform raw delta_t for decoder input
        decoder_input_delta_t_log = self.log_transform_fn(target_delta_t_raw, eps=self.log_eps)
        decoder_input_delta_t_log = decoder_input_delta_t_log * batch["attention_mask"].float()
        
        # Log-transform raw counts for decoder input
        decoder_input_counts_log = self.log_transform_fn(target_counts_raw, eps=self.log_eps)
        decoder_input_counts_log = decoder_input_counts_log * batch["attention_mask"].float()
        
        decoder_rnn_input_seq = torch.stack([decoder_input_delta_t_log, decoder_input_counts_log], dim=2)

        # 3. NTPP Decoder
        decoder_output_params = self.decoder(z, decoder_rnn_input_seq)
        
        return {
            "mu": mu,
            "log_var": log_var,
            "decoder_params": decoder_output_params,
            "target_delta_t_raw": target_delta_t_raw,
            "target_counts_raw": target_counts_raw,
            "attention_mask": batch["attention_mask"]
        }

    def _shared_step(self, batch, batch_idx, stage_name):
        outputs = self.forward(batch)
        
        mu = outputs["mu"]
        log_var = outputs["log_var"]
        decoder_params = outputs["decoder_params"] # This is a dict
        target_delta_t_raw = outputs["target_delta_t_raw"]
        target_counts_raw = outputs["target_counts_raw"]
        attention_mask = outputs["attention_mask"] # Shape (B, S)

        # Loss Calculation
        # 1. Reconstruction Loss (NLL)
        # NLL for inter-event times (Exponential distribution)
        # decoder_params["time_params"] is rate (lambda)

        nll_time = self.nll_exponential(
            rate=decoder_params["time_params"],
            target_times=target_delta_t_raw,
            mask=attention_mask
        )
        
        # NLL for charges/counts (Gaussian distribution)
        # decoder_params["charge_means"], decoder_params["charge_log_stds"]
        nll_charge = self.nll_gaussian(
            mean=decoder_params["charge_means"],
            log_std=decoder_params["charge_log_stds"],
            target_values=target_counts_raw,
            mask=attention_mask
        )
        
        reconstruction_loss = nll_time + nll_charge
        
        # 2. KL Divergence
        # Ensure kl_divergence_gaussian is correctly averaging over batch and summing over latent dim
        kl_loss = self.kl_divergence_gaussian(mu, log_var)
        
        # Total VAE Loss
        # beta_kl_weight is accessed via self.hparams as it was saved from model_options
        total_loss = reconstruction_loss + self.hparams.beta_kl_weight * kl_loss

        # Logging
        self.log(f"{stage_name}_loss", total_loss, on_step=(stage_name=="train"), on_epoch=True, prog_bar=True, logger=True, batch_size=target_delta_t_raw.size(0))
        self.log(f"{stage_name}_recon_loss", reconstruction_loss, on_step=(stage_name=="train"), on_epoch=True, logger=True, batch_size=target_delta_t_raw.size(0))
        self.log(f"{stage_name}_kl_loss", kl_loss, on_step=(stage_name=="train"), on_epoch=True, logger=True, batch_size=target_delta_t_raw.size(0))
        self.log(f"{stage_name}_nll_time", nll_time, on_step=False, on_epoch=True, logger=True, batch_size=target_delta_t_raw.size(0))
        self.log(f"{stage_name}_nll_charge", nll_charge, on_step=False, on_epoch=True, logger=True, batch_size=target_delta_t_raw.size(0))

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    # test_step can be similar if using the same logic
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Uses AdamW optimizer and CosineAnnealingLR scheduler as per template.
        Parameters are taken from self.hparams (saved from training_options in cfg).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr, # from training_options
            weight_decay=self.hparams.weight_decay # from training_options
        )
        
        # lr_schedule_t_max and lr_schedule_eta_min should be in training_options
        # If T_max is epochs, it should be self.hparams.epochs, but template used self.hparams.lr_schedule[0]
        # Let's assume training_options has lr_schedule_t_max and lr_schedule_eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.lr_schedule_t_max, # e.g., number of epochs or total steps
            eta_min=self.hparams.lr_schedule_eta_min # e.g., 0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Call scheduler step_fn after each epoch
                "frequency": 1
            }
        }

    @torch.no_grad()
    def generate_sequence(self, z_initial, max_time_horizon, max_events, start_of_sequence_features=None):
        """
        Generates a sequence of events autoregressively from the NTPPDecoder.

        Args:
            z_initial (Tensor): The initial latent vector, shape (batch_size, latent_dim).
                                Typically, batch_size will be 1 for single sequence generation.
            max_time_horizon (float): Maximum cumulative time for the generated sequence.
            max_events (int): Maximum number of events to generate.
            start_of_sequence_features (Tensor, optional): Features for the first RNN input.
                                                           Shape (batch_size, 1, event_input_dim).
                                                           Defaults to zeros if None.
                                                           event_input_dim is (log_delta_t, log_charge).

        Returns:
            generated_times (list of Tensors): List of generated absolute event times for each sequence in batch.
            generated_charges (list of Tensors): List of generated event charges for each sequence in batch.
        """
        self.eval() # Ensure model is in eval mode
        batch_size = z_initial.size(0)
        device = z_initial.device

        # Initialize hidden state for the decoder RNN from z_initial
        init_hidden_flat = self.decoder.fc_init_hidden(z_initial)
        current_hidden_state = init_hidden_flat.view(batch_size, self.decoder.rnn.num_layers, self.decoder.hidden_dim).permute(1, 0, 2).contiguous()

        # Lists to store generated events for each item in the batch
        batch_generated_times_abs = [[] for _ in range(batch_size)]
        batch_generated_charges = [[] for _ in range(batch_size)]
        
        # Current absolute time for each sequence in the batch
        current_abs_time = torch.zeros(batch_size, device=device)
        
        # Features for the first input to the RNN
        if start_of_sequence_features is None:
            # Default to zeros: (log_delta_t=log(0+eps), log_charge=log(0+eps)) or just zeros.
            # Using zeros directly for simplicity, assuming log_transform handles it or model learns.
            # A very small log_delta_t and log_charge.
            # For log_transform(0, eps), this would be log(eps).
            # Let's use features representing a "pseudo" event before the first actual one.
            # For example, log_transform(small_positive_val, eps)
            # A common choice is just zeros, and the model learns to interpret it.
            prev_event_features = torch.zeros(batch_size, 1, self.decoder.event_input_dim, device=device)
        else:
            prev_event_features = start_of_sequence_features.to(device)
            if prev_event_features.ndim == 2: # If (batch_size, event_input_dim)
                prev_event_features = prev_event_features.unsqueeze(1) # Make (batch_size, 1, event_input_dim)


        # Keep track of which sequences are still active (not met stopping criteria)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_generated_events = torch.zeros(batch_size, dtype=torch.long, device=device)

        for _ in range(max_events):
            if not active_mask.any(): # Stop if all sequences are done
                break

            # Get RNN output based on current hidden state and previous event features
            # rnn_input is (batch_size, 1, event_input_dim)
            rnn_output, next_hidden_state = self.decoder.rnn(prev_event_features, current_hidden_state)
            # rnn_output shape: (batch_size, 1, hidden_dim)
            
            current_hidden_state = next_hidden_state # Update hidden state for next step

            # Predict parameters for time and charge distributions
            time_param = F.softplus(self.decoder.fc_time_param(rnn_output.squeeze(1))) # rate (lambda), (B, 1)
            charge_mean = self.decoder.fc_charge_mean(rnn_output.squeeze(1))          # (B, 1)
            charge_log_std = self.decoder.fc_charge_log_std(rnn_output.squeeze(1))    # (B, 1)

            # Sample inter-event time (delta_t) from Exponential(rate)
            # For Exponential, E[X] = 1/rate. Sample by -log(U)/rate where U ~ Uniform(0,1)
            u_time = torch.rand(batch_size, 1, device=device)
            sampled_delta_t = -torch.log(u_time) / time_param.clamp(min=1e-9) # (B, 1), clamp rate for stability

            # Sample charge from Gaussian(mean, std)
            std_charge = torch.exp(charge_log_std)
            sampled_charge = torch.normal(charge_mean, std_charge) # (B, 1)
            # Potentially clip or post-process charge (e.g., ensure positive if it represents counts)
            # For this project, raw counts can be >0. If it's binned, it can be 0.
            # If Gaussian can produce negative, might need F.relu() or other transform if counts must be non-negative.
            # For now, use raw sampled_charge.

            # Update absolute times and store events for active sequences
            for i in range(batch_size):
                if active_mask[i]:
                    new_abs_time = current_abs_time[i] + sampled_delta_t[i].item()
                    
                    if new_abs_time >= max_time_horizon:
                        active_mask[i] = False
                        continue # Don't add this event, stop this sequence

                    current_abs_time[i] = new_abs_time
                    batch_generated_times_abs[i].append(new_abs_time)
                    # Assuming charge should be an integer count if it represents grouped hits
                    # and non-negative. For now, store float.
                    batch_generated_charges[i].append(sampled_charge[i].item())
                    num_generated_events[i] +=1
                    
                    if num_generated_events[i] >= max_events:
                        active_mask[i] = False


            # Prepare features for the next RNN input using the just-generated event
            # (log_transformed delta_t, log_transformed charge)
            # Need to handle potential zeros if sampled_delta_t or sampled_charge can be <=0 before log_transform
            # For delta_t from Exponential, it's > 0.
            # For charge from Gaussian, it can be anything. If charge must be >0 for log_transform:
            #   clamped_charge = torch.clamp(sampled_charge, min=self.log_eps) # Or some small positive value
            #   log_sampled_charge = self.log_transform_fn(clamped_charge, eps=self.log_eps)
            # For now, assume sampled_charge can be used with log_transform (e.g. if it's always positive or log_transform handles it)
            
            log_sampled_delta_t = self.log_transform_fn(sampled_delta_t.clamp(min=self.log_eps), eps=self.log_eps)
            # If charge represents counts, it should be positive. Let's assume it is for log_transform.
            # If it can be zero or negative, it needs careful handling (e.g. log1p or clamping).
            # For simplicity, let's assume it's positive for now.
            log_sampled_charge = self.log_transform_fn(sampled_charge.clamp(min=self.log_eps), eps=self.log_eps)
            
            prev_event_features = torch.cat([log_sampled_delta_t, log_sampled_charge], dim=1).unsqueeze(1)
            # prev_event_features shape: (batch_size, 1, event_input_dim)

        # Convert lists of tensors to simple tensors for easier handling if all have same length (after padding)
        # Or return lists of varying length tensors. Plan implies returning lists.
        final_times = [torch.tensor(times, device=device) for times in batch_generated_times_abs]
        final_charges = [torch.tensor(charges, device=device) for charges in batch_generated_charges]
        
        return final_times, final_charges