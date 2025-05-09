import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActNorm1d(nn.Module):
    """
    Activation Normalization for 1D inputs.
    """
    def __init__(self, num_features: int):
        super().__init__()
        if num_features != 1:
            raise ValueError("ActNorm1d currently only supports num_features=1.")
        self.num_features = num_features
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        
        # Learnable parameters for scale and bias
        # logs stores log(scale)
        self.logs = nn.Parameter(torch.zeros(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

    def _initialize(self, x: torch.Tensor):
        """
        Data-dependent initialization.
        """
        with torch.no_grad():
            mean = torch.mean(x, dim=0, keepdim=True)
            std = torch.std(x, dim=0, keepdim=True)
            
            self.logs.data = -torch.log(std + 1e-6)
            self.bias.data = -mean * torch.exp(self.logs.data)
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # x: (batch_size, num_features)
        if not self.initialized:
            self._initialize(x)

        outputs = torch.exp(self.logs) * x + self.bias
        log_abs_det_jacobian = self.logs.sum(dim=-1).expand(x.shape[0]) # (batch_size,)
        return outputs, log_abs_det_jacobian

    def inverse(self, y: torch.Tensor, context: torch.Tensor = None):
        # y: (batch_size, num_features)
        if not self.initialized:
            # Inverse might be called before forward during log_prob calculation.
            # Initialization should ideally happen with forward pass data.
            # For now, if called without init, it will use default zeros, which is not ideal.
            # A proper solution would require passing a batch for init or ensuring forward is called first.
            # However, Glow typically initializes on the first forward pass of training.
            pass # Rely on forward pass for initialization

        inputs = (y - self.bias) * torch.exp(-self.logs)
        log_abs_det_jacobian = -self.logs.sum(dim=-1).expand(y.shape[0]) # (batch_size,)
        return inputs, log_abs_det_jacobian


class InvertibleLinear1d(nn.Module):
    """
    Invertible Linear transformation (1x1 convolution for 1D features).
    """
    def __init__(self, num_features: int):
        super().__init__()
        if num_features != 1:
            raise ValueError("InvertibleLinear1d currently only supports num_features=1.")
        self.num_features = num_features
        
        # Weight matrix W (scalar for num_features=1)
        # Initialize to identity (1.0)
        self.weight = nn.Parameter(torch.randn(num_features, num_features))
        with torch.no_grad():
            # For num_features=1, this is just setting W to 1.0
            self.weight.copy_(torch.eye(num_features))


    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # x: (batch_size, num_features)
        # For num_features=1, self.weight is (1,1)
        outputs = F.linear(x, self.weight) # x @ W.T
        
        # log_abs_det_jacobian = torch.slogdet(self.weight)[1] # log|det(W)|
        # For num_features=1, det(W) = W[0,0]
        log_abs_det_jacobian = torch.log(torch.abs(self.weight[0,0]))
        return outputs, log_abs_det_jacobian.expand(x.shape[0])

    def inverse(self, y: torch.Tensor, context: torch.Tensor = None):
        # y: (batch_size, num_features)
        # For num_features=1, W_inv = 1/W[0,0]
        weight_inv = torch.inverse(self.weight)
        inputs = F.linear(y, weight_inv)
        
        # log_abs_det_jacobian = torch.slogdet(weight_inv)[1] # log|det(W_inv)| = -log|det(W)|
        log_abs_det_jacobian = -torch.log(torch.abs(self.weight[0,0]))
        return inputs, log_abs_det_jacobian.expand(y.shape[0])


class AffineConditionalTransformation1d(nn.Module):
    """
    Affine transformation conditioned on context.
    y = x * exp(s(context)) + t(context)
    """
    def __init__(self, num_features: int, context_dim: int, hidden_dims_s_t_net: list, activation_fn_str: str = "relu"):
        super().__init__()
        if num_features != 1:
            raise ValueError("AffineConditionalTransformation1d currently only supports num_features=1.")
        self.num_features = num_features
        
        s_t_net_layers = []
        current_dim = context_dim
        for hidden_dim in hidden_dims_s_t_net:
            s_t_net_layers.append(nn.Linear(current_dim, hidden_dim))
            if activation_fn_str == "relu":
                s_t_net_layers.append(nn.ReLU())
            elif activation_fn_str == "tanh":
                s_t_net_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation_fn_str}")
            current_dim = hidden_dim
        s_t_net_layers.append(nn.Linear(current_dim, num_features * 2)) # For s and t
        self.s_t_net = nn.Sequential(*s_t_net_layers)

    def _compute_s_t(self, context: torch.Tensor):
        # context: (batch_size, context_dim)
        s_t_params = self.s_t_net(context) # (batch_size, num_features * 2)
        s, t = torch.chunk(s_t_params, 2, dim=-1) # each (batch_size, num_features)
        # To ensure exp(s) is well-behaved, often s is scaled e.g. by tanh
        # s = torch.tanh(s) # Optional: constrain scale
        return s, t

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (batch_size, num_features)
        s, t = self._compute_s_t(context)
        outputs = x * torch.exp(s) + t
        log_abs_det_jacobian = s.sum(dim=-1) # (batch_size,)
        return outputs, log_abs_det_jacobian

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        # y: (batch_size, num_features)
        s, t = self._compute_s_t(context)
        inputs = (y - t) * torch.exp(-s)
        log_abs_det_jacobian = -s.sum(dim=-1) # (batch_size,)
        return inputs, log_abs_det_jacobian


class GlowStep1d(nn.Module):
    """
    A single step of Glow: ActNorm -> InvertibleLinear -> AffineConditionalTransformation.
    """
    def __init__(self, num_features: int, context_dim: int, hidden_dims_s_t_net: list, activation_fn_str: str = "relu"):
        super().__init__()
        self.actnorm = ActNorm1d(num_features)
        self.invlinear = InvertibleLinear1d(num_features)
        self.affine_cond = AffineConditionalTransformation1d(num_features, context_dim, hidden_dims_s_t_net, activation_fn_str)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        
        x, log_det = self.actnorm(x, context)
        total_log_det += log_det
        
        x, log_det = self.invlinear(x, context)
        total_log_det += log_det
        
        x, log_det = self.affine_cond(x, context)
        total_log_det += log_det
        
        return x, total_log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        total_log_det = torch.zeros(y.shape[0], device=y.device)
        
        y, log_det = self.affine_cond.inverse(y, context)
        total_log_det += log_det
        
        y, log_det = self.invlinear.inverse(y, context)
        total_log_det += log_det
        
        y, log_det = self.actnorm.inverse(y, context)
        total_log_det += log_det
        
        return y, total_log_det


class Glow1D(nn.Module):
    """
    Glow model for 1D inputs, conditioned on context.
    """
    def __init__(self, input_dim: int, context_dim: int, num_steps: int, 
                 hidden_dims_s_t_net: list, activation_s_t_net: str = "relu"):
        super().__init__()
        if input_dim != 1:
            raise ValueError("Glow1D currently only supports input_dim=1.")
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_steps = num_steps

        self.steps = nn.ModuleList([
            GlowStep1d(input_dim, context_dim, hidden_dims_s_t_net, activation_s_t_net)
            for _ in range(num_steps)
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
        
        for step in reversed(self.steps):
            current_inputs, log_det_j = step.inverse(current_inputs, context)
            log_det_jacobian_sum += log_det_j
            
        base_log_prob = self._base_distribution().log_prob(current_inputs)
        if self.input_dim > 1 : # Should not happen given constructor check, but for safety
             base_log_prob = base_log_prob.sum(dim=-1)
        else: # if input_dim is 1, log_prob output is (batch, 1), squeeze it
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
        
        # log_det_jacobian_sum = torch.zeros(current_samples.shape[0], device=context.device) # Not typically needed for sampling

        for step in self.steps:
            current_samples, _ = step.forward(current_samples, expanded_context) # log_det_j not needed for sampling
            
        # Reshape samples back to (batch_size, num_samples, input_dim)
        return current_samples.reshape(batch_size, num_samples, self.input_dim)