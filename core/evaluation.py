"""Evaluation Proxies for ATLAS-MERGE.

Calculates synthetic evaluation proxies grounded directly in the merged PyTorch tensors:
1. Mathematical Reasoning Proxy (GSM8K synthetic proxy)
2. Code Generation Proxy (MBPP synthetic proxy)
3. General Knowledge Retention Proxy (HellaSwag synthetic proxy)

IMPORTANT: These are synthetic research evaluation proxies derived from tensor
geometry and interference dynamics, NOT actual GSM8K, MBPP or HellaSwag benchmark measurements.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import math
import torch

from .synthetic_data import BlockTensors
from .geometry import GeometryProfile
from utils.validation import check_numerical_validity


@dataclass
class EvaluationResult:
    """Holds structured evaluation proxy metrics for a merged candidate."""
    math_proxy: float           # Synthetic GSM8K proxy in [0.0, 1.0]
    code_proxy: float           # Synthetic MBPP proxy in [0.0, 1.0]
    retention_proxy: float      # Synthetic HellaSwag retention proxy in [0.0, 1.0]
    block_math_scores: Dict[str, float]
    block_code_scores: Dict[str, float]
    block_retention_scores: Dict[str, float]
    is_valid: bool
    is_synthetic_proxy: bool = True
    disclaimer: str = "SYNTHETIC PROXY — NOT ACTUAL BENCHMARK RUN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationEngine:
    """Calculates proxy task retention and interference metrics across all blocks."""

    def __init__(
        self,
        base_math_floor: float = 0.20,
        base_code_floor: float = 0.22,
        max_task_gain: float = 0.70,
        distortion_sensitivity: float = 1.2,
    ):
        self.base_math_floor = base_math_floor
        self.base_code_floor = base_code_floor
        self.max_task_gain = max_task_gain
        self.distortion_sensitivity = distortion_sensitivity

    def evaluate_merged_model(
        self,
        blocks: List[BlockTensors],
        merged_tensors: Dict[str, torch.Tensor],
        geometry_profiles: Dict[str, GeometryProfile],
        recipes: Dict[str, Dict[str, Any]],
    ) -> EvaluationResult:
        """Evaluate the merged model against math, code, and retention proxies."""
        block_math: Dict[str, float] = {}
        block_code: Dict[str, float] = {}
        block_ret: Dict[str, float] = {}

        is_valid = True

        for block in blocks:
            name = block.name
            merged = merged_tensors.get(name)
            if merged is None or not check_numerical_validity(merged):
                is_valid = False
                break

            base_f = block.base.detach().reshape(-1).to(torch.float32)
            m_f = block.task_math.detach().reshape(-1).to(torch.float32)
            c_f = block.task_code.detach().reshape(-1).to(torch.float32)
            merged_f = merged.detach().reshape(-1).to(torch.float32)
            delta_f = merged_f - base_f

            norm_delta = torch.norm(delta_f, p=2).item()
            norm_m = torch.norm(m_f, p=2).item()
            norm_c = torch.norm(c_f, p=2).item()
            norm_base = torch.norm(base_f, p=2).item()

            if norm_delta < 1e-6:
                # BASE-PRESERVE: No task update applied
                s_m = 0.0
                s_c = 0.0
                s_r = 1.0  # Perfect base retention
            else:
                # Math alignment and magnitude preservation
                dot_m = torch.dot(delta_f, m_f).item()
                cos_m = max(0.0, dot_m / (norm_delta * norm_m + 1e-8))
                mag_ratio_m = min(1.0, norm_delta / (norm_m + 1e-8))
                s_m = cos_m * mag_ratio_m

                # Code alignment and magnitude preservation
                dot_c = torch.dot(delta_f, c_f).item()
                cos_c = max(0.0, dot_c / (norm_delta * norm_c + 1e-8))
                mag_ratio_c = min(1.0, norm_delta / (norm_c + 1e-8))
                s_c = cos_c * mag_ratio_c

                # Interference mitigation bonus/penalty
                # When sign conflict is high, naive task arithmetic suffers interference
                # TIES/DARE_TIES resolves sign interference
                geom = geometry_profiles.get(name)
                conflict = geom.sign_conflict_density if geom else 0.5
                op = recipes.get(name, {}).get("operator", "TASK_ARITHMETIC")

                if conflict > 0.5 and op == "TASK_ARITHMETIC":
                    interference_penalty = 0.15 * conflict
                    s_m = max(0.0, s_m - interference_penalty)
                    s_c = max(0.0, s_c - interference_penalty)

                # Base retention proxy (HellaSwag retention proxy)
                distortion = norm_delta / (norm_base + 1e-8)
                s_r = math.exp(-self.distortion_sensitivity * distortion)

            block_math[name] = float(s_m)
            block_code[name] = float(s_c)
            block_ret[name] = float(s_r)

        if not is_valid or len(blocks) == 0:
            return EvaluationResult(
                math_proxy=0.0,
                code_proxy=0.0,
                retention_proxy=0.0,
                block_math_scores={},
                block_code_scores={},
                block_retention_scores={},
                is_valid=False,
            )

        mean_math_gain = sum(block_math.values()) / len(blocks)
        mean_code_gain = sum(block_code.values()) / len(blocks)
        mean_retention = sum(block_ret.values()) / len(blocks)

        math_proxy = min(1.0, self.base_math_floor + self.max_task_gain * mean_math_gain)
        code_proxy = min(1.0, self.base_code_floor + self.max_task_gain * mean_code_gain)
        retention_proxy = min(1.0, max(0.0, mean_retention))

        return EvaluationResult(
            math_proxy=float(math_proxy),
            code_proxy=float(code_proxy),
            retention_proxy=float(retention_proxy),
            block_math_scores=block_math,
            block_code_scores=block_code,
            block_retention_scores=block_ret,
            is_valid=True,
        )

