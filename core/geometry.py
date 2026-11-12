"""Geometry Profiler for ATLAS-MERGE.

Calculates exact geometric descriptors between specialist task vectors:
1. Cosine similarity
2. Sign-conflict density
3. Relative L2 norm
4. Sparsity
"""
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import torch

from .synthetic_data import BlockTensors
from utils.validation import check_numerical_validity


@dataclass
class GeometryProfile:
    """Structured geometric profile for a functional block's task vectors."""
    block_name: str
    block_index: int
    cosine_similarity: float
    sign_conflict_density: float
    relative_norm: float
    sparsity_math: float
    sparsity_code: float
    sparsity_mean: float
    norm_math: float
    norm_code: float
    active_coordinate_ratio: float
    is_valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_state_vector(self) -> List[float]:
        """Return the 4 primary geometric descriptors for RL agent state."""
        return [
            float(self.cosine_similarity),
            float(self.sign_conflict_density),
            float(self.relative_norm),
            float(self.sparsity_mean),
        ]


def compute_cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Calculate mathematically exact cosine similarity: dot(A,B) / (norm(A)*norm(B) + eps)."""
    if not check_numerical_validity(tensor_a, tensor_b):
        return 0.0
    
    a_flat = tensor_a.detach().reshape(-1).to(torch.float32)
    b_flat = tensor_b.detach().reshape(-1).to(torch.float32)

    dot_product = torch.dot(a_flat, b_flat).item()
    norm_a = torch.norm(a_flat, p=2).item()
    norm_b = torch.norm(b_flat, p=2).item()

    denominator = (norm_a * norm_b) + epsilon
    cos_sim = dot_product / denominator
    # Bound within [-1.0, 1.0] to prevent floating point inaccuracies
    return float(max(-1.0, min(1.0, cos_sim)))


def compute_sign_conflict_density(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    magnitude_threshold: float = 1e-4,
    epsilon: float = 1e-8
) -> float:
    """Calculate fraction of active coordinates where task updates have opposing signs."""
    if not check_numerical_validity(tensor_a, tensor_b):
        return 0.0

    a_flat = tensor_a.detach().reshape(-1).to(torch.float32)
    b_flat = tensor_b.detach().reshape(-1).to(torch.float32)

    active_mask = (torch.abs(a_flat) > magnitude_threshold) & (torch.abs(b_flat) > magnitude_threshold)
    active_count = torch.sum(active_mask).item()

    if active_count == 0:
        return 0.0

    conflict_mask = active_mask & ((a_flat * b_flat) < 0.0)
    conflict_count = torch.sum(conflict_mask).item()

    density = conflict_count / (active_count + epsilon)
    return float(max(0.0, min(1.0, density)))


def compute_relative_l2_norm(tensor_a: torch.Tensor, tensor_b: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Calculate relative L2 norm: norm(B) / (norm(A) + eps)."""
    if not check_numerical_validity(tensor_a, tensor_b):
        return 1.0

    a_flat = tensor_a.detach().reshape(-1).to(torch.float32)
    b_flat = tensor_b.detach().reshape(-1).to(torch.float32)

    norm_a = torch.norm(a_flat, p=2).item()
    norm_b = torch.norm(b_flat, p=2).item()

    rel_norm = norm_b / (norm_a + epsilon)
    return float(rel_norm)


def compute_sparsity(tensor: torch.Tensor, threshold: float = 1e-3) -> float:
    """Calculate fraction of coordinates with absolute magnitude below threshold."""
    if not check_numerical_validity(tensor):
        return 0.0

    flat = tensor.detach().reshape(-1).to(torch.float32)
    total_elements = flat.numel()
    if total_elements == 0:
        return 0.0

    sparse_count = torch.sum(torch.abs(flat) <= threshold).item()
    return float(sparse_count / total_elements)


def profile_single_block(
    block: BlockTensors,
    magnitude_threshold: float = 1e-4,
    sparsity_threshold: float = 1e-3,
    epsilon: float = 1e-8,
) -> GeometryProfile:
    """Profile the task vector geometry of a single functional transformer block."""
    t_math = block.task_math
    t_code = block.task_code

    is_valid = check_numerical_validity(t_math, t_code, block.base)

    if not is_valid:
        return GeometryProfile(
            block_name=block.name,
            block_index=block.block_index,
            cosine_similarity=0.0,
            sign_conflict_density=0.0,
            relative_norm=1.0,
            sparsity_math=1.0,
            sparsity_code=1.0,
            sparsity_mean=1.0,
            norm_math=0.0,
            norm_code=0.0,
            active_coordinate_ratio=0.0,
            is_valid=False,
        )

    cos_sim = compute_cosine_similarity(t_math, t_code, epsilon)
    conflict_density = compute_sign_conflict_density(t_math, t_code, magnitude_threshold, epsilon)
    rel_norm = compute_relative_l2_norm(t_math, t_code, epsilon)
    sp_math = compute_sparsity(t_math, sparsity_threshold)
    sp_code = compute_sparsity(t_code, sparsity_threshold)
    sp_mean = 0.5 * (sp_math + sp_code)

    norm_m = float(torch.norm(t_math.detach().reshape(-1), p=2).item())
    norm_c = float(torch.norm(t_code.detach().reshape(-1), p=2).item())

    total_coords = t_math.numel()
    active_coords = torch.sum(
        (torch.abs(t_math.detach()) > magnitude_threshold) & (torch.abs(t_code.detach()) > magnitude_threshold)
    ).item()
    active_ratio = float(active_coords / total_coords) if total_coords > 0 else 0.0

    return GeometryProfile(
        block_name=block.name,
        block_index=block.block_index,
        cosine_similarity=cos_sim,
        sign_conflict_density=conflict_density,
        relative_norm=rel_norm,
        sparsity_math=sp_math,
        sparsity_code=sp_code,
        sparsity_mean=sp_mean,
        norm_math=norm_m,
        norm_code=norm_c,
        active_coordinate_ratio=active_ratio,
        is_valid=True,
    )


def profile_all_blocks(
    blocks: List[BlockTensors],
    magnitude_threshold: float = 1e-4,
    sparsity_threshold: float = 1e-3,
    epsilon: float = 1e-8,
) -> Dict[str, GeometryProfile]:
    """Profile all eight functional transformer blocks independently."""
    profiles: Dict[str, GeometryProfile] = {}
    for block in blocks:
        profile = profile_single_block(
            block=block,
            magnitude_threshold=magnitude_threshold,
            sparsity_threshold=sparsity_threshold,
            epsilon=epsilon,
        )
        profiles[block.name] = profile
    return profiles


def save_geometry_profiles(profiles: Dict[str, GeometryProfile], filepath: str) -> str:
    """Save geometric profiles to a structured JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable = {name: prof.to_dict() for name, prof in profiles.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return os.path.abspath(filepath)


def load_geometry_profiles(filepath: str) -> Dict[str, GeometryProfile]:
    """Load geometric profiles from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    profiles = {}
    for name, d in raw_data.items():
        profiles[name] = GeometryProfile(**d)
    return profiles

