import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
from torch.distributions import LogNormal, Normal
from utils.utils import log_transform, inverse_log_transform

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Changed to [1, max_len, d_model] for easier broadcasting
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True)
        """
        # x is (batch_size, seq_len, d_model)
        # self.pe is (1, max_len, d_model)
        # We need to add pe[:, :x.size(1), :] to x
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Om2vecModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        model_options = self.hparams['model_options']
        data_options = self.hparams['data_options']

        d_model = model_options['d_model']
        n_head = model_options['n_head']
        dim_feedforward = model_options['dim_feedforward']
        dropout_rate = model_options['dropout'] # Renamed from 'dropout' to avoid conflict with nn.Dropout module
        
        self.max_seq_len = data_options['max_seq_len']

        # Input embedding: projects 2 features (time, charge) to d_model
        # This will be used for both encoder input and decoder input (target sequence)
        self.feature_embed = nn.Linear(2, d_model) 
        
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_len=self.max_seq_len)

        # ---- Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate, 
            batch_first=True,
            activation='relu' # Standard activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_options['encoder_n_layers'])
        
        self.learned_latent_dim = model_options['latent_dim'] - 1
        if self.learned_latent_dim <= 0:
            raise ValueError("model_options['latent_dim'] must be > 1 for the learned part.")
        
        self.fc_mu_rest = nn.Linear(d_model, self.learned_latent_dim)
        self.fc_logvar_rest = nn.Linear(d_model, self.learned_latent_dim)

        # ---- Decoder (NTPP) ----
        self.latent_to_memory_transform = nn.Linear(model_options['latent_dim'], d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate, 
            batch_first=True,
            activation='relu' # Standard activation
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=model_options['decoder_n_layers'])

        self.fc_delta_t_loc = nn.Linear(d_model, 1)
        self.fc_delta_t_scale = nn.Linear(d_model, 1)
        self.fc_charge_loc = nn.Linear(d_model, 1)
        self.fc_charge_scale = nn.Linear(d_model, 1)

        self.kl_beta_start = model_options.get('kl_beta_start', 1.0)
        self.kl_beta_end = model_options.get('kl_beta_end', 1.0)
        self.kl_anneal_epochs = model_options.get('kl_anneal_epochs', 0)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, input_tq_sequence, attention_mask):
        # input_tq_sequence: (B, L, 2) [norm_abs_t, norm_abs_q]
        # attention_mask: (B, L) boolean mask, True for valid entries, False for padding
        
        src = self.feature_embed(input_tq_sequence) # (B, L, d_model)
        src = src * math.sqrt(self.hparams['model_options']['d_model'])
        src = self.pos_encoder(src)

        # src_key_padding_mask: (B, L), True for padded elements (inverse of attention_mask)
        src_key_padding_mask = ~attention_mask 

        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # encoder_output shape: (B, L, d_model)
        
        # Use the output of the first element (like a CLS token) for VAE parameters
        # This assumes the first token aggregates sequence information.
        # Alternative: mean pooling over non-padded elements.
        # pooled_output = encoder_output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_output = encoder_output[:, 0, :] # (B, d_model)

        mu_rest = self.fc_mu_rest(pooled_output)
        logvar_rest = self.fc_logvar_rest(pooled_output)
        return mu_rest, logvar_rest

    def reparameterize(self, mu_rest, logvar_rest):
        std_rest = torch.exp(0.5 * logvar_rest)
        eps_rest = torch.randn_like(std_rest)
        return mu_rest + eps_rest * std_rest

    def decode(self, z, decoder_input_sequence, tgt_key_padding_mask):
        # z: (B, total_latent_dim) - full latent vector
        # decoder_input_sequence: (B, L_tgt, 2) [shifted_norm_delta_t, shifted_norm_charge] for teacher forcing
        # tgt_key_padding_mask: (B, L_tgt) boolean, True for padded elements in decoder_input_sequence
        
        seq_len_tgt = decoder_input_sequence.size(1)

        memory_latent = self.latent_to_memory_transform(z) # (B, d_model)
        # Expand memory to be (B, 1, d_model) to act as a single memory item for the decoder to attend to.
        # The TransformerDecoder will attend to this single memory vector across all target positions.
        memory = memory_latent.unsqueeze(1) # (B, 1, d_model)
        # No memory_key_padding_mask needed as memory is a single, always valid item.

        tgt = self.feature_embed(decoder_input_sequence) # (B, L_tgt, d_model)
        tgt = tgt * math.sqrt(self.hparams['model_options']['d_model'])
        tgt = self.pos_encoder(tgt)

        tgt_mask = self._generate_square_subsequent_mask(seq_len_tgt, device=z.device) # (L_tgt, L_tgt)

        decoder_output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None # Memory is a single item, always unpadded
        )
        # decoder_output shape: (B, L_tgt, d_model)
            
        dt_locs = self.fc_delta_t_loc(decoder_output)
        dt_scales = torch.exp(self.fc_delta_t_scale(decoder_output))
        charge_locs = self.fc_charge_loc(decoder_output)
        charge_scales = torch.exp(self.fc_charge_scale(decoder_output))
        
        return dt_locs, dt_scales, charge_locs, charge_scales

    def forward(self, batch):
        input_tq_sequence = batch["input_tq_sequence"] 
        attention_mask = batch["attention_mask"]      
        active_entries_count = batch["active_entries_count"].float()

        z0_true_log_len = torch.log(active_entries_count + 1.0).unsqueeze(1)

        mu_rest, logvar_rest = self.encode(input_tq_sequence, attention_mask)
        z_rest = self.reparameterize(mu_rest, logvar_rest)
        
        z = torch.cat([z0_true_log_len, z_rest], dim=1)

        abs_norm_t = input_tq_sequence[:, :, 0:1]
        norm_q = input_tq_sequence[:, :, 1:2]
        delta_norm_t = torch.zeros_like(abs_norm_t)
        delta_norm_t[:, 0, :] = abs_norm_t[:, 0, :]
        if self.max_seq_len > 1:
            delta_norm_t[:, 1:, :] = abs_norm_t[:, 1:, :] - abs_norm_t[:, :-1, :]
        
        decoder_target_for_loss = torch.cat([delta_norm_t, norm_q], dim=2)

        start_token_features = torch.zeros(input_tq_sequence.size(0), 1, 2, device=input_tq_sequence.device)
        decoder_input_for_decode = torch.cat([start_token_features, decoder_target_for_loss[:, :-1, :]], dim=1)
        
        # tgt_key_padding_mask for decoder_input_for_decode
        # If true_lengths_for_loss is L_true (number of actual events in target_for_loss),
        # then decoder_input_for_decode has L_true valid entries (start_token + L_true-1 items).
        # The mask should be True for padded elements.
        true_lengths_for_input = torch.round(torch.exp(z0_true_log_len.squeeze(dim=-1)) - 1.0).long()
        true_lengths_for_input = torch.clamp(true_lengths_for_input, min=1, max=self.max_seq_len)
        # The decoder_input_for_decode has `true_lengths_for_input` valid elements.
        # So, elements from index `true_lengths_for_input` onwards are padding.
        tgt_key_padding_mask = torch.arange(self.max_seq_len, device=z.device)[None, :] >= true_lengths_for_input[:, None]


        dt_locs, dt_scales, charge_locs, charge_scales = self.decode(z, decoder_input_for_decode, tgt_key_padding_mask)
        
        return dt_locs, dt_scales, charge_locs, charge_scales, mu_rest, logvar_rest, decoder_target_for_loss, z0_true_log_len

    def _shared_step(self, batch, batch_idx, stage_name):
        dt_locs, dt_scales, charge_locs, charge_scales, mu_rest, logvar_rest, target_sequence, z0_true_log_len = self.forward(batch)
        
        target_delta_t = target_sequence[:, :, 0:1]
        target_charge = target_sequence[:, :, 1:2]

        true_lengths_for_loss = torch.round(torch.exp(z0_true_log_len.squeeze(dim=-1)) - 1.0).long()
        true_lengths_for_loss = torch.clamp(true_lengths_for_loss, min=1, max=self.max_seq_len)

        loss_mask_seq = torch.arange(self.max_seq_len, device=target_sequence.device)[None, :] < true_lengths_for_loss[:, None]
        loss_mask = loss_mask_seq.unsqueeze(-1) 

        dist_dt = LogNormal(dt_locs, dt_scales.clamp(min=1e-6))
        log_prob_dt = dist_dt.log_prob(target_delta_t.clamp(min=1e-6)) 
        
        dist_charge = Normal(charge_locs, charge_scales.clamp(min=1e-6))
        log_prob_charge = dist_charge.log_prob(target_charge)

        masked_log_prob_dt = log_prob_dt * loss_mask
        masked_log_prob_charge = log_prob_charge * loss_mask
        
        active_elements_per_batch = loss_mask.sum()
        
        recon_loss_dt = -masked_log_prob_dt.sum() / active_elements_per_batch.clamp(min=1)
        recon_loss_charge = -masked_log_prob_charge.sum() / active_elements_per_batch.clamp(min=1)
        reconstruction_loss = recon_loss_dt + recon_loss_charge

        kl_divergence = -0.5 * torch.sum(1 + logvar_rest - mu_rest.pow(2) - logvar_rest.exp(), dim=1).mean()
        
        if self.kl_anneal_epochs > 0 and self.trainer.current_epoch < self.kl_anneal_epochs: # Use self.trainer.current_epoch
            kl_beta = self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * (self.trainer.current_epoch / self.kl_anneal_epochs)
        elif self.kl_anneal_epochs > 0 and self.trainer.current_epoch >= self.kl_anneal_epochs:
            kl_beta = self.kl_beta_end
        else: 
            kl_beta = self.kl_beta_end 

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
        if "lr_schedule" in opt_cfg and opt_cfg["lr_schedule"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opt_cfg["lr_schedule"][0], eta_min=opt_cfg["lr_schedule"][1]
            )
            return [optimizer], [scheduler]
        return optimizer

    @torch.no_grad()
    def generate(self, num_samples: int, z_vectors_provided: torch.Tensor, device: torch.device):
        self.eval()
        d_model = self.hparams['model_options']['d_model']
        
        all_generated_sequences_t_norm = []
        all_generated_sequences_q_norm = []

        for i in range(num_samples):
            z_sample_i = z_vectors_provided[i:i+1, :].to(device) 

            n_steps_for_this_sample = torch.round(torch.exp(z_sample_i[:, 0]) - 1.0).int().item()
            n_steps_for_this_sample = max(1, min(n_steps_for_this_sample, self.max_seq_len))

            memory_latent_i = self.latent_to_memory_transform(z_sample_i) 
            memory = memory_latent_i.unsqueeze(1) # (1, 1, d_model) - Global memory context

            # Start with a start-of-sequence token (batch_size=1, seq_len=1, features=2)
            # Features are (delta_t, charge), initially zeros.
            generated_tokens_features = torch.zeros(1, 1, 2, device=device) 
            
            current_generated_t_single_norm = []
            current_generated_q_single_norm = []

            for _ in range(n_steps_for_this_sample):
                # Embed current sequence of generated tokens
                embedded_tgt = self.feature_embed(generated_tokens_features) # (1, current_len, d_model)
                embedded_tgt = embedded_tgt * math.sqrt(d_model)
                embedded_tgt = self.pos_encoder(embedded_tgt)

                tgt_mask_gen = self._generate_square_subsequent_mask(embedded_tgt.size(1), device)
                
                decoder_output_step = self.transformer_decoder(
                    embedded_tgt, 
                    memory, 
                    tgt_mask=tgt_mask_gen
                    # No tgt_key_padding_mask needed as we build the sequence one by one without padding
                )
                
                last_token_output = decoder_output_step[:, -1:, :] # (1, 1, d_model)

                dt_loc = self.fc_delta_t_loc(last_token_output)
                dt_scale = torch.exp(self.fc_delta_t_scale(last_token_output))
                charge_loc = self.fc_charge_loc(last_token_output)
                charge_scale = torch.exp(self.fc_charge_scale(last_token_output))

                next_delta_t_norm = LogNormal(dt_loc, dt_scale.clamp(min=1e-6)).sample() 
                next_charge_norm = Normal(charge_loc, charge_scale.clamp(min=1e-6)).sample()
                
                current_generated_t_single_norm.append(next_delta_t_norm.squeeze().clone()) # Use .clone()
                current_generated_q_single_norm.append(next_charge_norm.squeeze().clone())

                next_feature_pair = torch.cat([next_delta_t_norm, next_charge_norm], dim=2) # (1,1,2)
                generated_tokens_features = torch.cat([generated_tokens_features, next_feature_pair], dim=1)
            
            if current_generated_t_single_norm:
                all_generated_sequences_t_norm.append(torch.stack(current_generated_t_single_norm))
                all_generated_sequences_q_norm.append(torch.stack(current_generated_q_single_norm))
            else:
                all_generated_sequences_t_norm.append(torch.tensor([], device=device))
                all_generated_sequences_q_norm.append(torch.tensor([], device=device))
        
        return all_generated_sequences_t_norm, all_generated_sequences_q_norm