"""Merge Engine for ATLAS-MERGE.

Orchestrates block-wise merging across all eight functional transformer blocks,
supporting heterogeneous operator selection, variable density, and variable scaling per block.
"""
from typing import Dict, List, Any, Optional
import torch

from .synthetic_data import BlockTensors
from .merge_operators import execute_merge_operator, OPERATOR_NAMES
from utils.validation import check_numerical_validity


class MergeEngine:
    """Executes full multi-block merges according to assigned recipes."""

    def __init__(self, discrete_densities: Optional[List[float]] = None, discrete_scales: Optional[List[float]] = None):
        self.discrete_densities = discrete_densities or [0.2, 0.4, 0.7, 1.0]
        self.discrete_scales = discrete_scales or [0.25, 0.5, 0.75, 1.0]

    def merge_blocks(
        self,
        blocks: List[BlockTensors],
        recipe: Dict[str, Dict[str, Any]],
        random_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge all eight blocks according to the block-wise recipe.
        
        recipe format:
        {
            "Q1 Attention": {"operator": "TIES", "density": 0.4, "scale": 0.75},
            ...
        }
        """
        merged_tensors: Dict[str, torch.Tensor] = {}

        for block in blocks:
            name = block.name
            block_cfg = recipe.get(name, {
                "operator": "TASK_ARITHMETIC",
                "density": 1.0,
                "scale": 1.0,
            })

            op = block_cfg.get("operator", "TASK_ARITHMETIC")
            density = float(block_cfg.get("density", 1.0))
            scale = float(block_cfg.get("scale", 1.0))

            task_vectors = [block.task_math, block.task_code]
            merged = execute_merge_operator(
                operator_name=op,
                base=block.base,
                task_vectors=task_vectors,
                density=density,
                scale=scale,
                random_seed=random_seed,
            )
            merged_tensors[name] = merged

        return merged_tensors

    def compute_merge_statistics(
        self,
        block: BlockTensors,
        merged_tensor: torch.Tensor,
        recipe_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute tensor statistics comparing base, specialist tasks, and the merged result."""
        base_f = block.base.detach().reshape(-1).to(torch.float32)
        m_f = block.task_math.detach().reshape(-1).to(torch.float32)
        c_f = block.task_code.detach().reshape(-1).to(torch.float32)
        merged_f = merged_tensor.detach().reshape(-1).to(torch.float32)
        delta_f = merged_f - base_f

        norm_base = float(torch.norm(base_f, p=2).item())
        norm_delta = float(torch.norm(delta_f, p=2).item())
        norm_merged = float(torch.norm(merged_f, p=2).item())

        # Alignment of delta with math and code updates
        norm_m = float(torch.norm(m_f, p=2).item())
        norm_c = float(torch.norm(c_f, p=2).item())

        cos_delta_math = float((torch.dot(delta_f, m_f) / (norm_delta * norm_m + 1e-8)).item()) if norm_delta > 1e-6 else 0.0
        cos_delta_code = float((torch.dot(delta_f, c_f) / (norm_delta * norm_c + 1e-8)).item()) if norm_delta > 1e-6 else 0.0

        sparsity_delta = float((torch.sum(torch.abs(delta_f) < 1e-4).item()) / delta_f.numel())

        return {
            "block_name": block.name,
            "operator": recipe_entry.get("operator", "UNKNOWN"),
            "density": float(recipe_entry.get("density", 1.0)),
            "scale": float(recipe_entry.get("scale", 1.0)),
            "norm_base": norm_base,
            "norm_delta": norm_delta,
            "norm_merged": norm_merged,
            "mean_merged": float(torch.mean(merged_f).item()),
            "std_merged": float(torch.std(merged_f).item()),
            "min_merged": float(torch.min(merged_f).item()),
            "max_merged": float(torch.max(merged_f).item()),
            "cos_delta_math": cos_delta_math,
            "cos_delta_code": cos_delta_code,
            "sparsity_delta": sparsity_delta,
            "is_numerically_valid": check_numerical_validity(merged_tensor),
        }

