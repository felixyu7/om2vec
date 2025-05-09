import torch
import matplotlib.pyplot as plt
import numpy as np
# Assuming Om2vecModel will be importable when this is used in a script
# from models.om2vec_model import Om2vecModel # Causes circular import if model imports this
from utils.om_processing import calculate_summary_statistics # To get z_summary

def plot_P_t_given_z(model, om_raw_times: torch.Tensor, om_raw_charges: torch.Tensor, 
                       om_sensor_pos: torch.Tensor, 
                       num_time_points: int = 200, time_range_padding: float = 0.1,
                       true_times_color: str = 'orange', pdf_color: str = 'blue'):
    """
    Plots the learned probability density P(t|z) for a single OM,
    optionally overlaying the true observed photon arrival times.

    Args:
        model: The trained Om2vecModel instance.
        om_raw_times (torch.Tensor): 1D tensor of raw (unnormalized, unpadded) photon arrival times for the OM.
        om_raw_charges (torch.Tensor): 1D tensor of raw charges corresponding to om_raw_times.
        om_sensor_pos (torch.Tensor): 1D tensor of the OM's sensor position [x,y,z].
        num_time_points (int): Number of points to evaluate the PDF P(t|z) over.
        time_range_padding (float): Padding factor for the time range of the plot, relative to data range.
        true_times_color (str): Color for plotting true photon arrival times.
        pdf_color (str): Color for plotting the learned PDF.
    """
    model.eval() # Ensure model is in evaluation mode
    device = next(model.parameters()).device

    om_raw_times = om_raw_times.to(device).float()
    om_raw_charges = om_raw_charges.to(device).float()
    om_sensor_pos = om_sensor_pos.to(device).float()

    if om_raw_times.numel() == 0:
        print("Warning: No photon times provided for plotting P(t|z). Skipping plot.")
        return

    with torch.no_grad():
        # 1. Calculate z_summary
        z_summary = calculate_summary_statistics(
            om_raw_times, 
            om_raw_charges,
            charge_log_offset=model.data_cfg.get('charge_log_offset', 1.0),
            time_log_epsilon=model.data_cfg.get('time_log_epsilon', 1e-9)
        ).unsqueeze(0) # Add batch dim (N_valid_oms=1)

        # 2. Preprocess (t,q) for encoder (normalize, pad - though padding won't affect single OM much here)
        # We need to create a dummy batch structure similar to what the model's forward expects for a single OM.
        # This part is a bit tricky as the model.forward() is designed for batches of events.
        # For plotting, we are interested in a single OM's P(t|z).
        # We'll simulate the encoder path for this single OM.
        
        # Normalize times and charges for encoder input
        norm_times = (om_raw_times - model.data_cfg.get('time_norm_mean', 0.0)) / \
                      model.data_cfg.get('time_norm_std', 1.0)
        norm_charges = torch.log(om_raw_charges + model.data_cfg.get('charge_log_offset', 1.0))
        
        normalized_tq_sequence_unpadded = torch.stack((norm_times, norm_charges), dim=-1) # (N_hits, 2)
        
        # Pad for transformer
        num_hits = normalized_tq_sequence_unpadded.shape[0]
        max_photons = model.data_cfg['max_photons_per_om']
        
        padded_tq_sequence = torch.zeros((1, max_photons, 2), device=device) # (1, P, 2)
        current_hit_mask = torch.zeros((1, max_photons), dtype=torch.bool, device=device) # (1, P)

        num_hits_to_take = min(num_hits, max_photons)
        if num_hits_to_take > 0:
            padded_tq_sequence[0, :num_hits_to_take, :] = normalized_tq_sequence_unpadded[:num_hits_to_take, :]
            current_hit_mask[0, :num_hits_to_take] = True
            
        # 3. Encoder pass to get z_learned (using mu_learned for deterministic plot)
        embedded_tq = model.input_embedder(padded_tq_sequence) # (1, P, D_emb)
        embedded_tq += model.pos_encoder.to(device)[:max_photons, :] # Add positional encoding

        transformer_input = embedded_tq
        if model.model_cfg.get('sensor_integration_type') == 'concat_to_transformer_input' and \
           model.sensor_pos_embedder is not None:
            sensor_pos_norm_scale = model.data_cfg.get('sensor_pos_norm_scale', 1.0)
            norm_sensor_pos = om_sensor_pos.unsqueeze(0) / sensor_pos_norm_scale # (1, 3)
            sensor_pos_emb = model.sensor_pos_embedder(norm_sensor_pos) # (1, D_sensor_emb)
            sensor_pos_emb_expanded = sensor_pos_emb.unsqueeze(1).expand(-1, max_photons, -1)
            transformer_input = torch.cat((embedded_tq, sensor_pos_emb_expanded), dim=-1)

        transformer_padding_mask = ~current_hit_mask
        transformer_output = model.transformer_encoder(transformer_input, src_key_padding_mask=transformer_padding_mask)

        # Pooling
        if num_hits_to_take == 0: # Handle OM with no hits after padding/truncation
             pooled_output = torch.zeros(1, transformer_output.shape[-1], device=device)
        elif model.model_cfg.get('pooling_strategy', 'mean') == 'mean':
            expanded_hit_mask_plot = current_hit_mask.unsqueeze(-1).expand_as(transformer_output)
            summed_output = (transformer_output * expanded_hit_mask_plot).sum(dim=1)
            num_actual_hits_plot = current_hit_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            pooled_output = summed_output / num_actual_hits_plot
        else: # Should not happen based on current config
            raise ValueError(f"Unsupported pooling for plotting: {model.model_cfg.get('pooling_strategy')}")

        mu_learned = model.fc_mu_learned(pooled_output) # (1, D_learned)
        # For plotting, we use mu_learned directly (no sampling)
        z_learned_deterministic = mu_learned

        # 4. Form full z and context for CNF
        z_full = torch.cat((z_summary, z_learned_deterministic), dim=1) # (1, 9 + D_learned)
        
        cnf_context = z_full
        if model.cnf_conditions_on_sensor_pos:
            sensor_pos_norm_scale = model.data_cfg.get('sensor_pos_norm_scale', 1.0)
            norm_sensor_pos_for_cnf = om_sensor_pos.unsqueeze(0) / sensor_pos_norm_scale
            sensor_pos_emb_for_cnf = model.sensor_pos_embedder(norm_sensor_pos_for_cnf)
            cnf_context = torch.cat((z_full, sensor_pos_emb_for_cnf), dim=1)

        # 5. Generate a range of t values for plotting the PDF
        min_t = torch.min(om_raw_times)
        max_t = torch.max(om_raw_times)
        padding = (max_t - min_t) * time_range_padding
        if padding == 0: # Handle case with single hit or all hits at same time
            padding = max_t * time_range_padding if max_t > 0 else 1.0 # Default padding
            
        t_plot_range_min = min_t - padding
        t_plot_range_max = max_t + padding
        
        # Ensure range is sensible if min_t == max_t
        if t_plot_range_min >= t_plot_range_max:
            t_plot_range_min = min_t - 1.0 # Default range around the point
            t_plot_range_max = max_t + 1.0

        t_values_for_pdf = torch.linspace(t_plot_range_min, t_plot_range_max, num_time_points, device=device).unsqueeze(-1) # (num_time_points, 1)

        # Repeat context for each t_value
        cnf_context_repeated = cnf_context.expand(num_time_points, -1) # (num_time_points, context_dim)

        # 6. Compute P(t|z) using CNF decoder's log_prob and then exp()
        log_probs_pdf = model.cnf_decoder.log_prob(t_values_for_pdf, context=cnf_context_repeated)
        probs_pdf = torch.exp(log_probs_pdf)

        # 7. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(t_values_for_pdf.cpu().numpy(), probs_pdf.cpu().numpy(), label=r'$P(t|z)$', color=pdf_color, linewidth=2)
        
        # Overlay true photon arrival times as a rug plot or histogram
        if om_raw_times.numel() > 0:
            # Normalize charges for sizing markers in rug plot, or use for histogram weights
            # For simplicity, using fixed alpha/size for rug plot points
            plt.plot(om_raw_times.cpu().numpy(), np.zeros_like(om_raw_times.cpu().numpy()) - 0.01 * torch.max(probs_pdf).item(), 
                     '|', color=true_times_color, markersize=10, alpha=0.7, label='Observed Photons')
            # Alternative: histogram
            # plt.hist(om_raw_times.cpu().numpy(), bins=50, density=True, alpha=0.5, label='Observed Photon Density', color=true_times_color)


        plt.xlabel('Photon Arrival Time (t)')
        plt.ylabel('Probability Density P(t|z)')
        plt.title(f'Learned Photon Arrival Time Distribution for OM')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=min(-0.02 * torch.max(probs_pdf).item(), -0.01)) # Ensure rug plot markers are visible
        plt.show()

if __name__ == '__main__':
    # This block is for testing the plotting function.
    # It requires a dummy model and dummy OM data.
    # For a full test, one would need to instantiate Om2vecModel from om2vec_model.py
    # This will be tricky due to circular dependencies if Om2vecModel imports this.
    # Better to test this from a separate script or notebook after model is stable.

    print("utils/plotting.py executed. Contains plot_P_t_given_z.")
    print("To test plotting, call plot_P_t_given_z with an instantiated model and OM data from a separate script.")
    
    # Example (conceptual, won't run directly here without model def):
    # from models.om2vec_model import Om2vecModel # Assuming this can be resolved
    # dummy_cfg = { ... } # A full config for Om2vecModel
    # model_instance = Om2vecModel(dummy_cfg)
    # dummy_times = torch.tensor([100., 110., 150., 160., 200.])
    # dummy_charges = torch.tensor([1., 0.5, 2., 1.2, 0.8])
    # dummy_pos = torch.tensor([10., 20., 30.])
    # plot_P_t_given_z(model_instance, dummy_times, dummy_charges, dummy_pos)