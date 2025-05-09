import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AffineCouplingLayer1d(nn.Module):
    """
    An affine coupling layer for 1D inputs, conditioned on context.
    This is the core component of a RealNVP-style flow for 1D data.
    y = x * exp(s(context)) + t(context)
    """
    def __init__(self, input_dim: int, context_dim: int, hidden_dims_s_t_net: list, activation_fn_str: str = "relu"):
        super().__init__()
        if input_dim != 1:
            raise ValueError("AffineCouplingLayer1d currently only supports input_dim=1.")
        self.input_dim = input_dim # Should be 1
        
        s_t_net_layers = []
        current_dim = context_dim
        for hidden_dim in hidden_dims_s_t_net:
            s_t_net_layers.append(nn.Linear(current_dim, hidden_dim))
            if activation_fn_str.lower() == "relu":
                s_t_net_layers.append(nn.ReLU())
            elif activation_fn_str.lower() == "tanh":
                s_t_net_layers.append(nn.Tanh())
            elif activation_fn_str.lower() == "leaky_relu":
                s_t_net_layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation_fn_str}")
            current_dim = hidden_dim
        # Output is 2 * input_dim because we need s and t for each dimension (here, just 1 dim).
        s_t_net_layers.append(nn.Linear(current_dim, self.input_dim * 2))
        self.s_t_net = nn.Sequential(*s_t_net_layers)

        # Initialize the final layer to output near-zero scales and zero shifts initially
        # This helps stabilize training at the beginning.
        # The s_t_net outputs s_raw and t. We want exp(s_raw) to be close to 1.
        # So s_raw should be close to 0.
        with torch.no_grad():
            last_layer = self.s_t_net[-1]
            last_layer.weight.data.normal_(0, 0.01) # Small weights for s_raw and t
            last_layer.bias.data.zero_()


    def _compute_s_t(self, context: torch.Tensor):
        # context: (batch_size, context_dim)
        s_t_params = self.s_t_net(context) # (batch_size, input_dim * 2)
        # For input_dim=1, s_raw and t are each (batch_size, 1)
        s_raw, t = torch.chunk(s_t_params, 2, dim=-1)
        
        # To make exp(s_raw) more stable, often s_raw is scaled, e.g., by a tanh.
        # This keeps the scale factor from exploding or vanishing too quickly.
        # s = torch.tanh(s_raw) # Example: scale s_raw to [-1, 1]
        # Or a learnable scale factor for s_raw
        # For simplicity, we'll use s_raw directly but the above initialization helps.
        return s_raw, t

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (batch_size, input_dim) which is (batch_size, 1)
        s_raw, t = self._compute_s_t(context) # s_raw, t are (batch_size, 1)
        
        # Apply a non-linearity to s_raw to control the scale factor, e.g., tanh
        # This helps prevent exp(s_raw) from becoming too large or too small.
        # Let's use a scaling factor to keep s values in a reasonable range for exp.
        log_scale = torch.tanh(s_raw) # s_raw is now in [-1, 1], so exp(s_raw) is in [~0.36, ~2.7]
                                      # Or, more commonly, s is output directly and used as log_scale.
                                      # Let's assume s_raw is the log_scale directly.
        
        outputs = x * torch.exp(s_raw) + t
        log_abs_det_jacobian = s_raw.sum(dim=-1) # (batch_size,) as s_raw is (batch_size, 1)
        return outputs, log_abs_det_jacobian

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        # y: (batch_size, input_dim) which is (batch_size, 1)
        s_raw, t = self._compute_s_t(context) # s_raw, t are (batch_size, 1)
        
        # log_scale = torch.tanh(s_raw) # Consistent with forward
        
        inputs = (y - t) * torch.exp(-s_raw)
        log_abs_det_jacobian = -s_raw.sum(dim=-1) # (batch_size,)
        return inputs, log_abs_det_jacobian


class RealNVP1D(nn.Module):
    """
    RealNVP-style flow for 1D inputs, conditioned on context.
    Composed of a sequence of AffineCouplingLayer1d.
    """
    def __init__(self, input_dim: int, context_dim: int, num_coupling_layers: int, 
                 hidden_dims_s_t_net: list, activation_s_t_net: str = "relu"):
        super().__init__()
        if input_dim != 1:
            raise ValueError("RealNVP1D currently only supports input_dim=1.")
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_coupling_layers = num_coupling_layers

        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer1d(input_dim, context_dim, hidden_dims_s_t_net, activation_s_t_net)
            for _ in range(num_coupling_layers)
        ])
        
        # Base distribution (standard normal)
        self.register_buffer('base_loc', torch.zeros(input_dim))
        self.register_buffer('base_scale', torch.ones(input_dim))

    def _base_distribution(self):
        return torch.distributions.Normal(self.base_loc, self.base_scale)

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor):
        """
        Computes the log probability of inputs under the flow.
        Args:
            inputs (torch.Tensor): Shape (batch_size, input_dim)
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            torch.Tensor: Log probabilities, shape (batch_size,)
        """
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {inputs.shape[-1]}")

        log_det_jacobian_sum = torch.zeros(inputs.shape[0], device=inputs.device)
        current_inputs = inputs
        
        # Inverse pass: transform data to base distribution
        for layer in reversed(self.coupling_layers):
            current_inputs, log_det_j = layer.inverse(current_inputs, context)
            log_det_jacobian_sum += log_det_j
            
        base_log_prob = self._base_distribution().log_prob(current_inputs)
        # For input_dim=1, log_prob output is (batch_size, 1), squeeze it.
        base_log_prob = base_log_prob.squeeze(-1)

        return base_log_prob + log_det_jacobian_sum

    def sample(self, num_samples: int, context: torch.Tensor):
        """
        Samples from the flow.
        Args:
            num_samples (int): Number of samples to generate for each context item.
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            torch.Tensor: Samples, shape (batch_size, num_samples, input_dim)
        """
        batch_size = context.shape[0]
        
        # Samples from base distribution: (batch_size, num_samples, input_dim)
        base_samples = self._base_distribution().sample((batch_size, num_samples))
        
        # Reshape for processing: (batch_size * num_samples, input_dim)
        current_samples = base_samples.reshape(-1, self.input_dim)
        
        # Expand context: (batch_size, context_dim) -> (batch_size * num_samples, context_dim)
        expanded_context = context.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.context_dim)
        
        # Forward pass: transform base samples to data space
        for layer in self.coupling_layers:
            current_samples, _ = layer.forward(current_samples, expanded_context) # log_det_j not needed for sampling
            
        # Reshape samples back to (batch_size, num_samples, input_dim)
        return current_samples.reshape(batch_size, num_samples, self.input_dim)