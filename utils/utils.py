import torch

def log_transform(tensor_data: torch.Tensor) -> torch.Tensor:
    """
    Applies log(tensor_data + 1.0) element-wise.
    Handles potential zeros by adding 1 before logging.
    """
    return torch.log(tensor_data + 1.0)

def inverse_log_transform(tensor_data: torch.Tensor) -> torch.Tensor:
    """
    Applies exp(tensor_data) - 1.0 element-wise.
    Inverse of log_transform.
    """
    return torch.exp(tensor_data) - 1.0

# Placeholder for other potential NTPP-related PDF/Sampling utilities
# For now, we will rely on torch.distributions