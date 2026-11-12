"""Merge Operators for ATLAS-MERGE.

Implements mathematically rigorous model merge operators:
1. BASE-PRESERVE: Retains unmodified base parameters.
2. TASK ARITHMETIC: Linear combination of task vectors added to base.
3. TIES: Trimming, Sign Election, and Disjoint Sign-Compatible Aggregation.
4. DARE-TIES: Drop And REscale with TIES aggregation.
5. SLERP: Spherical Linear Interpolation (optional comparative baseline).
"""
from typing import Tuple, List, Optional
import torch
from utils.validation import check_numerical_validity

OPERATOR_NAMES = ["BASE_PRESERVE", "TASK_ARITHMETIC", "TIES", "DARE_TIES"]


def base_preserve_merge(base: torch.Tensor) -> torch.Tensor:
    """BASE-PRESERVE operator: returns the unmodified base parameters."""
    return base.clone()


def task_arithmetic_merge(
    base: torch.Tensor,
    task_vectors: List[torch.Tensor],
    scale: float = 1.0,
    task_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """TASK ARITHMETIC operator:
    merged = base + scale * sum_i(weight_i * task_i)
    """
    if not task_vectors:
        return base.clone()

    if task_weights is None:
        # Default equal weighting normalized
        task_weights = [1.0 / len(task_vectors)] * len(task_vectors)
    
    total_update = torch.zeros_like(base, dtype=torch.float32)
    for w, t in zip(task_weights, task_vectors):
        total_update = total_update + (float(w) * t.to(torch.float32))

    merged = base.to(torch.float32) + (float(scale) * total_update)
    return merged


def ties_merge(
    base: torch.Tensor,
    task_vectors: List[torch.Tensor],
    density: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    """TIES (Trim, Elect Sign, & Merge) operator.
    
    Steps:
    1. Trimming: Keep top (100 * density)% coordinates by magnitude per task vector.
    2. Sign Election: Determine coordinate-wise majority sign across trimmed updates.
    3. Sign-Compatible Aggregation: Retain only updates matching the elected sign.
    4. Addition to base scaled by lambda.
    """
    if not task_vectors:
        return base.clone()

    # Clamp density within legal interval (0.0, 1.0]
    density = max(1e-4, min(1.0, float(density)))
    scale = float(scale)

    flat_shape = base.shape
    num_coords = base.numel()
    k = max(1, min(num_coords, int(round(density * num_coords))))

    trimmed_vectors: List[torch.Tensor] = []

    # Step 1: Magnitude Trimming
    for t in task_vectors:
        t_flat = t.detach().reshape(-1).to(torch.float32)
        if density < 1.0:
            abs_vals = torch.abs(t_flat)
            # Find threshold value for top-k magnitude
            topk_vals, _ = torch.topk(abs_vals, k=k, largest=True, sorted=False)
            threshold = topk_vals.min()
            # Retain coordinates >= threshold
            mask = abs_vals >= threshold
            trimmed_flat = t_flat * mask.float()
        else:
            trimmed_flat = t_flat.clone()
        trimmed_vectors.append(trimmed_flat)

    # Step 2: Sign Election (Majority voting across trimmed task vectors)
    # Sum of trimmed coordinates
    stacked_trimmed = torch.stack(trimmed_vectors, dim=0) # [M, num_coords]
    summed_update = torch.sum(stacked_trimmed, dim=0)       # [num_coords]
    elected_sign = torch.sign(summed_update)                # {-1, 0, +1}

    # Step 3: Sign-compatible aggregation
    # For each task vector, keep coordinate only if its sign matches elected_sign
    signs_match = (torch.sign(stacked_trimmed) == elected_sign.unsqueeze(0)) & (stacked_trimmed != 0.0)
    compatible_updates = stacked_trimmed * signs_match.float()

    # Mean over compatible coordinates
    compat_counts = torch.sum(signs_match.float(), dim=0) # [num_coords]
    has_compat = compat_counts > 0

    agg_update = torch.zeros(num_coords, dtype=torch.float32, device=base.device)
    sum_compat = torch.sum(compatible_updates, dim=0)
    agg_update[has_compat] = sum_compat[has_compat] / (compat_counts[has_compat] + 1e-8)

    # Step 4: Add to base
    merged = base.to(torch.float32) + (scale * agg_update.reshape(flat_shape))
    return merged


def dare_ties_merge(
    base: torch.Tensor,
    task_vectors: List[torch.Tensor],
    density: float = 0.5,
    scale: float = 1.0,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """DARE-TIES (Drop And REscale with TIES) operator.
    
    1. Randomly drop coordinates with drop rate (1 - density), keep probability p = density.
    2. Rescale kept coordinates by 1 / p.
    3. Execute TIES aggregation on the rescaled task vectors.
    """
    if not task_vectors:
        return base.clone()

    density = max(1e-4, min(1.0, float(density)))
    p_keep = density

    generator = None
    if random_seed is not None:
        generator = torch.Generator(device=base.device).manual_seed(random_seed)

    rescaled_vectors: List[torch.Tensor] = []
    for t in task_vectors:
        t_float = t.detach().to(torch.float32)
        if p_keep < 1.0:
            if generator is not None:
                mask = (torch.rand(t_float.shape, generator=generator, device=base.device) < p_keep).float()
            else:
                mask = (torch.rand(t_float.shape, device=base.device) < p_keep).float()
            # Inverse keep probability rescaling
            rescaled = (t_float * mask) / p_keep
        else:
            rescaled = t_float.clone()
        rescaled_vectors.append(rescaled)

    # Pass the DARE-rescaled vectors to TIES merge
    return ties_merge(
        base=base,
        task_vectors=rescaled_vectors,
        density=density,
        scale=scale,
    )


def slerp_merge(
    base: torch.Tensor,
    task1: torch.Tensor,
    task2: torch.Tensor,
    t: float = 0.5,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """SLERP (Spherical Linear Interpolation) baseline between two specialist models."""
    v0 = (base + task1).detach().reshape(-1).to(torch.float32)
    v1 = (base + task2).detach().reshape(-1).to(torch.float32)

    norm_v0 = torch.norm(v0, p=2) + epsilon
    norm_v1 = torch.norm(v1, p=2) + epsilon

    u0 = v0 / norm_v0
    u1 = v1 / norm_v1

    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0).item()

    if abs(dot) > 0.9995:
        # Near colinear, use linear interpolation
        interpolated = (1.0 - t) * v0 + t * v1
    else:
        theta = torch.acos(torch.tensor(dot, dtype=torch.float32))
        sin_theta = torch.sin(theta)
        w0 = torch.sin((1.0 - t) * theta) / sin_theta
        w1 = torch.sin(t * theta) / sin_theta
        # Retain interpolated magnitude
        target_norm = (1.0 - t) * norm_v0 + t * norm_v1
        u_interp = w0 * u0 + w1 * u1
        interpolated = u_interp * target_norm

    return interpolated.reshape(base.shape)


def execute_merge_operator(
    operator_name: str,
    base: torch.Tensor,
    task_vectors: List[torch.Tensor],
    density: float = 1.0,
    scale: float = 1.0,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """Dispatcher to execute a specific merge operator with given hyperparameters."""
    op_upper = operator_name.upper().strip()
    if op_upper == "BASE_PRESERVE" or op_upper == "0":
        return base_preserve_merge(base)
    elif op_upper == "TASK_ARITHMETIC" or op_upper == "1":
        return task_arithmetic_merge(base, task_vectors, scale=scale)
    elif op_upper == "TIES" or op_upper == "2":
        return ties_merge(base, task_vectors, density=density, scale=scale)
    elif op_upper == "DARE_TIES" or op_upper == "3":
        return dare_ties_merge(base, task_vectors, density=density, scale=scale, random_seed=random_seed)
    elif op_upper == "SLERP":
        if len(task_vectors) >= 2:
            return slerp_merge(base, task_vectors[0], task_vectors[1], t=0.5)
        elif len(task_vectors) == 1:
            return base + scale * task_vectors[0]
        return base.clone()
    else:
        raise ValueError(f"Unknown merge operator: {operator_name}. Expected one of {OPERATOR_NAMES} or SLERP.")

