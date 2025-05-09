import torch
import torch.nn as nn
import numpy as np

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class RationalQuadraticSpline(nn.Module):
    """
    A layer implementing a rational-quadratic spline transformation.
    This will be the core building block.
    """
    def __init__(self, input_dim: int, context_dim: int, num_bins: int, hidden_dims_hypernet: list):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_bins = num_bins
        
        if input_dim != 1:
            raise ValueError("This simplified RationalQuadraticSpline currently only supports 1D inputs.")

        # Hypernetwork to generate spline parameters from context
        # It needs to output:
        # - unnormalized_widths (num_bins)
        # - unnormalized_heights (num_bins)
        # - unnormalized_derivatives (num_bins + 1)
        self.num_outputs_hypernet = num_bins + num_bins + (num_bins + 1)
        
        layers = []
        current_dim = context_dim
        for hidden_dim in hidden_dims_hypernet:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU()) # Using ReLU as a common simple activation
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.num_outputs_hypernet))
        self.hypernet = nn.Sequential(*layers)

        # Define bounds for the spline transformation, e.g., [-bound, bound]
        # These are often set based on the expected range of the data or base distribution.
        self.left = -5.0
        self.right = 5.0
        self.bottom = -5.0
        self.top = 5.0


    def _compute_and_normalize_spline_params(self, context: torch.Tensor):
        """
        Computes spline parameters from context and normalizes them.
        Args:
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            Tuple of tensors: x_pos, y_pos, derivatives
                              Shapes: (batch_size, num_bins+1), (batch_size, num_bins+1), (batch_size, num_bins+1)
        """
        batch_size = context.shape[0]
        params = self.hypernet(context) # (batch_size, num_outputs_hypernet)
        
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins : 2 * self.num_bins]
        # Derivatives at internal knots + 2 boundary derivatives
        unnormalized_derivatives = params[..., 2 * self.num_bins:]

        # Ensure widths and heights are positive using softmax
        # Add a fixed value for stability if all are zero, then normalize
        widths = nn.functional.softmax(unnormalized_widths, dim=-1) * (self.right - self.left)
        heights = nn.functional.softmax(unnormalized_heights, dim=-1) * (self.top - self.bottom)

        # Ensure derivatives are positive using softplus
        derivatives = nn.functional.softplus(unnormalized_derivatives) + DEFAULT_MIN_DERIVATIVE

        # Calculate knot positions (x_pos, y_pos)
        # x_pos are cumulative sums of widths, starting from self.left
        x_pos = torch.zeros(batch_size, self.num_bins + 1, device=context.device)
        x_pos[..., 0] = self.left
        x_pos[..., 1:] = self.left + torch.cumsum(widths, dim=-1)
        x_pos[..., -1] = self.right # Ensure last knot is exactly at the boundary

        # y_pos are cumulative sums of heights, starting from self.bottom
        y_pos = torch.zeros(batch_size, self.num_bins + 1, device=context.device)
        y_pos[..., 0] = self.bottom
        y_pos[..., 1:] = self.bottom + torch.cumsum(heights, dim=-1)
        y_pos[..., -1] = self.top # Ensure last knot is exactly at the boundary
        
        return x_pos, y_pos, derivatives

    def _rational_quadratic_spline_transformation(self, inputs, x_pos, y_pos, derivatives, inverse=False):
        """
        Core rational-quadratic spline transformation logic.
        Can perform forward or inverse transformation.
        Based on Durkan et al., 2019 (Neural Spline Flows).
        """
        batch_size, num_points = inputs.shape[0], inputs.shape[1] # inputs are (batch_size, 1) for 1D
        
        # Reshape inputs to (batch_size) for easier indexing if it's (batch_size, 1)
        if inputs.shape[-1] == 1:
            inputs_flat = inputs.squeeze(-1) # (batch_size)
        else:
            inputs_flat = inputs # Should be (batch_size)

        # Find which bin each input falls into
        # For inverse, inputs are y-values, search in y_pos
        # For forward, inputs are x-values, search in x_pos
        knot_positions = y_pos if inverse else x_pos # (batch_size, num_bins+1)
        
        # Searchsorted needs sorted arrays. Knot positions should be monotonic.
        # We need to find k such that knot_positions[k] <= input < knot_positions[k+1]
        # `torch.searchsorted` returns the index where the element would be inserted to maintain order.
        # If input is knot_positions[k], it returns k. We want k.
        # If input is between knot_positions[k] and knot_positions[k+1], it returns k+1. We want k.
        # So, we subtract 1 from the result of searchsorted, and clamp at 0.
        bin_indices = torch.searchsorted(knot_positions, inputs_flat.unsqueeze(-1), right=False).squeeze(-1) -1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1) # (batch_size)

        # Select the parameters for the identified bins
        # Gather expects indices to be same shape as output, or broadcastable
        # bin_indices needs to be (batch_size, 1) for gather
        bin_idx_expanded = bin_indices.unsqueeze(-1) # (batch_size, 1)

        # x_k, y_k for the left of the bin
        x_k = torch.gather(x_pos, 1, bin_idx_expanded).squeeze(-1) # (batch_size)
        y_k = torch.gather(y_pos, 1, bin_idx_expanded).squeeze(-1) # (batch_size)
        
        # x_{k+1}, y_{k+1} for the right of the bin
        x_k_plus_1 = torch.gather(x_pos, 1, bin_idx_expanded + 1).squeeze(-1)
        y_k_plus_1 = torch.gather(y_pos, 1, bin_idx_expanded + 1).squeeze(-1)
        
        # Derivatives d_k, d_{k+1}
        d_k = torch.gather(derivatives, 1, bin_idx_expanded).squeeze(-1)
        d_k_plus_1 = torch.gather(derivatives, 1, bin_idx_expanded + 1).squeeze(-1)

        # Bin widths and heights
        bin_width = x_k_plus_1 - x_k
        bin_height = y_k_plus_1 - y_k

        # Slope s_k = (y_{k+1} - y_k) / (x_{k+1} - x_k)
        s_k = bin_height / bin_width

        if inverse:
            # Inverse transformation: solve for xi given y
            # xi = (y - y_k) / (y_{k+1} - y_k)
            # This is relative position within the *output* bin
            xi_numerator = inputs_flat - y_k
            xi_denominator = bin_height
            # Avoid division by zero if bin_height is tiny (should be prevented by min_bin_height)
            xi = xi_numerator / (xi_denominator + 1e-9) # Add epsilon for stability
            xi = torch.clamp(xi, 0, 1) # Ensure xi is in [0,1]

            # Common term for quadratic solution
            # (d_{k+1} + d_k - 2*s_k) * xi * (1-xi)
            common_term_numerator = (d_k_plus_1 + d_k - 2 * s_k) * xi * (1 - xi)
            common_term_denominator = s_k + (d_k_plus_1 + d_k - 2 * s_k) * xi
            # Add epsilon to denominator for stability
            common_term = common_term_numerator / (common_term_denominator + 1e-9)
            
            # Output x = x_k + (x_{k+1} - x_k) * [s_k * xi^2 + d_k * xi * (1-xi)] / [s_k + (d_{k+1} + d_k - 2*s_k) * xi * (1-xi)]
            # This simplifies to: x = x_k + bin_width * [ (s_k - d_k)*xi^2 + d_k*xi ] / [ s_k + (d_{k+1} + d_k - 2*s_k)*xi*(1-xi) ]
            # No, the formula from paper is:
            # x = x_k + bin_width * [ (s_k * xi^2) + (d_k * xi * (1-xi)) ] / [ s_k + (d_{k+1} + d_k - 2*s_k) * xi * (1-xi) ]
            # This is equivalent to: x = x_k + bin_width * xi * [s_k*xi + d_k*(1-xi)] / [s_k + (d_{k+1} + d_k - 2*s_k)*xi*(1-xi)]
            
            # Let's use the form: x = x_k + bin_width * [ (s_k - d_k)*xi^2 + d_k*xi ] / denominator_for_x
            # where denominator_for_x = s_k + (d_{k+1} + d_k - 2*s_k)*xi*(1-xi)
            # This is not quite right. The paper's formula for y(xi) is:
            # y(xi) = y_k + (y_{k+1} - y_k) * [ s_k * xi^2 + d_k * xi * (1-xi) ] / [ s_k + (d_{k+1} + d_k - 2*s_k) * xi * (1-xi) ]
            # where xi = (x - x_k) / (x_{k+1} - x_k)
            # For the inverse, we are given y, find xi_y = (y - y_k) / (y_{k+1} - y_k)
            # Then we need to solve for xi_x from:
            # xi_y = [ s_k * xi_x^2 + d_k * xi_x * (1-xi_x) ] / [ s_k + (d_{k+1} + d_k - 2*s_k) * xi_x * (1-xi_x) ]
            # This is a quadratic equation in xi_x.
            # Let A = s_k - d_k
            # Let B = d_k
            # Let C = s_k
            # Let D = d_{k+1} + d_k - 2*s_k
            # xi_y = [ A*xi_x^2 + B*xi_x ] / [ C + D*xi_x*(1-xi_x) ]
            # xi_y * [ C + D*xi_x - D*xi_x^2 ] = A*xi_x^2 + B*xi_x
            # xi_y*C + xi_y*D*xi_x - xi_y*D*xi_x^2 = A*xi_x^2 + B*xi_x
            # (A + xi_y*D)*xi_x^2 + (B - xi_y*D)*xi_x - xi_y*C = 0
            # This is a_quad * xi_x^2 + b_quad * xi_x + c_quad = 0
            a_quad = (s_k - d_k) + xi * (d_k_plus_1 + d_k - 2*s_k) # Here xi is xi_y
            b_quad = d_k - xi * (d_k_plus_1 + d_k - 2*s_k)         # Here xi is xi_y
            c_quad = -s_k * xi                                    # Here xi is xi_y

            # Solve quadratic: xi_x = [-b_quad +/- sqrt(b_quad^2 - 4*a_quad*c_quad)] / (2*a_quad)
            discriminant = b_quad.pow(2) - 4 * a_quad * c_quad
            # Ensure discriminant is non-negative
            discriminant = torch.clamp(discriminant, min=0)
            
            # Numerator: -b_quad + sqrt(discriminant) because xi_x should be in [0,1] and increasing
            # For RQ splines, the solution with +sqrt is usually the correct one for the principal branch.
            sol_numerator = -b_quad + torch.sqrt(discriminant)
            sol_denominator = 2 * a_quad
            
            # Handle cases where a_quad is close to zero (linear segment)
            # If a_quad is zero, then b_quad * xi_x + c_quad = 0 => xi_x = -c_quad / b_quad
            is_linear = torch.abs(a_quad) < 1e-7
            xi_x = torch.where(
                is_linear,
                -c_quad / (b_quad + 1e-9), # Add epsilon for stability
                sol_numerator / (sol_denominator + 1e-9) # Add epsilon for stability
            )
            xi_x = torch.clamp(xi_x, 0, 1) # Ensure result is in [0,1]

            outputs = x_k + bin_width * xi_x
            
        else: # Forward transformation
            # xi = (x - x_k) / (x_{k+1} - x_k)
            # This is relative position within the *input* bin
            xi_numerator = inputs_flat - x_k
            xi_denominator = bin_width
            xi = xi_numerator / (xi_denominator + 1e-9) # Add epsilon for stability
            xi = torch.clamp(xi, 0, 1) # Ensure xi is in [0,1]

            # y(xi) = y_k + (y_{k+1} - y_k) * [ s_k * xi^2 + d_k * xi * (1-xi) ] / [ s_k + (d_{k+1} + d_k - 2*s_k) * xi * (1-xi) ]
            numerator_y = s_k * xi.pow(2) + d_k * xi * (1 - xi)
            denominator_y = s_k + (d_k_plus_1 + d_k - 2 * s_k) * xi * (1 - xi)
            
            outputs = y_k + bin_height * (numerator_y / (denominator_y + 1e-9)) # Add epsilon

        # Log absolute determinant of Jacobian
        # log |dy/dx| = log | (dy/d_xi) / (dx/d_xi) |
        # dy/dx = [ s_k^2 * (d_{k+1}*xi^2 + 2*s_k*xi*(1-xi) + d_k*(1-xi)^2) ] / [ s_k + (d_{k+1} + d_k - 2*s_k)*xi*(1-xi) ]^2
        # This is dy/dx at point x (which corresponds to xi)
        
        # Numerator of derivative: s_k^2 * (d_{k+1}*xi^2 + 2*s_k*xi*(1-xi) + d_k*(1-xi)^2)
        # Denominator of derivative: [ s_k + (d_{k+1} + d_k - 2*s_k)*xi*(1-xi) ]^2
        
        # Let common_term_for_derivative_denom = s_k + (d_{k+1} + d_k - 2*s_k)*xi*(1-xi)
        # This is `denominator_y` from the forward pass if `inverse` is false.
        # If `inverse` is true, `xi` is `xi_y`, we need `xi_x` for the derivative.
        # The derivative is always dy/dx, so we need xi based on x.
        # If inverse=True, we calculated xi_x. Use that.
        # If inverse=False, we calculated xi. Use that.
        current_xi_for_derivative = xi_x if inverse else xi

        term_in_derivative_num = d_k_plus_1 * current_xi_for_derivative.pow(2) + \
                                 2 * s_k * current_xi_for_derivative * (1 - current_xi_for_derivative) + \
                                 d_k * (1 - current_xi_for_derivative).pow(2)
        
        derivative_numerator = s_k.pow(2) * term_in_derivative_num
        
        derivative_denominator_term = s_k + (d_k_plus_1 + d_k - 2 * s_k) * \
                                      current_xi_for_derivative * (1 - current_xi_for_derivative)
        derivative_denominator = derivative_denominator_term.pow(2)
        
        log_abs_det_jacobian = torch.log(derivative_numerator + 1e-9) - torch.log(derivative_denominator + 1e-9)
        # Ensure log_abs_det_jacobian is not NaN if derivative_numerator is zero.
        # derivative_numerator should be positive due to positive s_k and derivatives.
        
        # Reshape outputs back to (batch_size, 1) if original input was
        if inputs.shape[-1] == 1:
            outputs = outputs.unsqueeze(-1)
            
        return outputs, log_abs_det_jacobian


    def forward(self, inputs: torch.Tensor, context: torch.Tensor):
        """
        Apply the forward transformation.
        Args:
            inputs (torch.Tensor): Shape (batch_size, input_dim) which is (batch_size, 1)
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                outputs (torch.Tensor): Shape (batch_size, input_dim)
                log_abs_det_jacobian (torch.Tensor): Shape (batch_size,)
        """
        x_pos, y_pos, derivatives = self._compute_and_normalize_spline_params(context)
        outputs, log_abs_det_jacobian = self._rational_quadratic_spline_transformation(
            inputs, x_pos, y_pos, derivatives, inverse=False
        )
        return outputs, log_abs_det_jacobian

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor):
        """
        Apply the inverse transformation.
        Args:
            inputs (torch.Tensor): Shape (batch_size, input_dim) which is (batch_size, 1)
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                outputs (torch.Tensor): Shape (batch_size, input_dim)
                log_abs_det_jacobian (torch.Tensor): Shape (batch_size,)
        """
        x_pos, y_pos, derivatives = self._compute_and_normalize_spline_params(context)
        outputs, log_abs_det_jacobian = self._rational_quadratic_spline_transformation(
            inputs, x_pos, y_pos, derivatives, inverse=True
        )
        return outputs, log_abs_det_jacobian


class NeuralSplineFlow(nn.Module):
    """
    A simple Neural Spline Flow model composed of multiple RationalQuadraticSpline layers.
    """
    def __init__(self, input_dim: int, context_dim: int, num_layers: int, 
                 num_bins: int, hidden_dims_hypernet: list):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            RationalQuadraticSpline(input_dim, context_dim, num_bins, hidden_dims_hypernet)
            for _ in range(num_layers)
        ])
        
        # Base distribution (e.g., standard normal)
        self.base_distribution = torch.distributions.Normal(
            loc=torch.zeros(input_dim), 
            scale=torch.ones(input_dim)
        )

    def _get_base_dist_params(self, device):
        return self.base_distribution.loc.to(device), self.base_distribution.scale.to(device)

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor):
        """
        Computes the log probability of inputs under the flow.
        Args:
            inputs (torch.Tensor): Shape (batch_size, input_dim)
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            torch.Tensor: Log probabilities, shape (batch_size,)
        """
        log_det_jacobian_sum = torch.zeros(inputs.shape[0], device=inputs.device)
        current_inputs = inputs
        
        # Inverse pass: transform data to base distribution
        for layer in reversed(self.layers):
            current_inputs, log_det_j = layer.inverse(current_inputs, context)
            log_det_jacobian_sum += log_det_j
            
        # Log probability in base space
        loc, scale = self._get_base_dist_params(inputs.device)
        base_log_prob = torch.distributions.Normal(loc, scale).log_prob(current_inputs).sum(dim=-1) # Sum over input_dim if > 1
        
        return base_log_prob + log_det_jacobian_sum

    def sample(self, num_samples: int, context: torch.Tensor):
        """
        Samples from the flow.
        Args:
            num_samples (int): Number of samples to generate for each context.
            context (torch.Tensor): Shape (batch_size, context_dim)
        Returns:
            torch.Tensor: Samples, shape (batch_size, num_samples, input_dim)
        """
        batch_size = context.shape[0]
        loc, scale = self._get_base_dist_params(context.device)
        
        # Samples from base distribution
        # Shape: (batch_size, num_samples, input_dim)
        base_samples = torch.distributions.Normal(loc, scale).sample((batch_size, num_samples))
        
        # Reshape context for broadcasting: (batch_size, 1, context_dim)
        # Reshape base_samples for layer processing: (batch_size * num_samples, input_dim)
        current_samples = base_samples.view(-1, self.input_dim)
        
        # Expand context to match the flattened samples
        # (batch_size, context_dim) -> (batch_size, 1, context_dim) -> (batch_size, num_samples, context_dim)
        # -> (batch_size * num_samples, context_dim)
        expanded_context = context.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.context_dim)
        
        log_det_jacobian_sum = torch.zeros(current_samples.shape[0], device=context.device)

        # Forward pass: transform base samples to data space
        for layer in self.layers:
            current_samples, log_det_j = layer.forward(current_samples, expanded_context)
            log_det_jacobian_sum += log_det_j # Not typically needed for sampling, but good to have
            
        # Reshape samples back to (batch_size, num_samples, input_dim)
        return current_samples.view(batch_size, num_samples, self.input_dim)