"""Validation and numerical integrity utilities for ATLAS-MERGE."""
from typing import Tuple, Any
import torch
import numpy as np

def validate_tensor(tensor: Any, expected_shape: Tuple[int, ...] = None, name: str = "tensor") -> Tuple[bool, str]:
    """Validate that an input is a valid PyTorch FP32 tensor with no NaNs or Infs."""
    if not isinstance(tensor, torch.Tensor):
        return False, f"{name} is not a torch.Tensor (got {type(tensor)})."
    
    if tensor.dtype != torch.float32:
        return False, f"{name} must have dtype torch.float32 (got {tensor.dtype})."
    
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        return False, f"{name} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}."
    
    if torch.isnan(tensor).any().item():
        return False, f"{name} contains NaN values."
    
    if torch.isinf(tensor).any().item():
        return False, f"{name} contains Inf values."
    
    return True, "Valid"


def check_numerical_validity(*values: Any) -> bool:
    """Return True if all provided scalars/tensors are finite and non-NaN."""
    for val in values:
        if val is None:
            return False
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return False
        elif isinstance(val, torch.Tensor):
            if torch.isnan(val).any().item() or torch.isinf(val).any().item():
                return False
        elif isinstance(val, np.ndarray):
            if np.isnan(val).any() or np.isinf(val).any():
                return False
    return True


def validate_action_indices(operator_idx: int, density_idx: int, scale_idx: int,
                            num_operators: int = 4, num_densities: int = 4, num_scales: int = 4) -> bool:
    """Validate that categorical action indices are within legal discrete ranges."""
    if not (0 <= operator_idx < num_operators):
        return False
    if not (0 <= density_idx < num_densities):
        return False
    if not (0 <= scale_idx < num_scales):
        return False
    return True

