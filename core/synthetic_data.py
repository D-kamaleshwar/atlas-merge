"""Synthetic Task-Vector Generation for ATLAS-MERGE.

Generates deterministic, mathematically grounded synthetic specialist task vectors
for eight functional transformer blocks.
NOTE: These are synthetic research simulation tensors, NOT actual Qwen model weights.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import torch

FUNCTIONAL_BLOCK_NAMES = [
    "Q1 Attention",
    "Q1 MLP",
    "Q2 Attention",
    "Q2 MLP",
    "Q3 Attention",
    "Q3 MLP",
    "Q4 Attention",
    "Q4 MLP",
]

@dataclass
class BlockTensors:
    """Holds actual PyTorch tensors for a single functional transformer block."""
    name: str
    block_index: int
    base: torch.Tensor          # Base model parameter tensor Theta_base
    task_math: torch.Tensor     # Math reasoning task update tau_math
    task_code: torch.Tensor     # Code generation task update tau_code
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_device(self, device: str) -> "BlockTensors":
        return BlockTensors(
            name=self.name,
            block_index=self.block_index,
            base=self.base.to(device),
            task_math=self.task_math.to(device),
            task_code=self.task_code.to(device),
            metadata=self.metadata.copy(),
        )


class SyntheticTaskVectorGenerator:
    """Generates synthetic base and task update tensors with controlled geometric properties.
    
    Ensures rich variation across eight functional blocks:
    - Directionally aligned blocks (e.g. Q1 Attention)
    - Conflicting / opposing blocks (e.g. Q2 MLP, Q4 MLP)
    - Orthogonal blocks (e.g. Q2 Attention)
    - Sparse blocks (e.g. Q4 Attention)
    """

    def __init__(
        self,
        tensor_shape: Tuple[int, int] = (128, 128),
        random_seed: int = 42,
        base_std: float = 0.02,
        task_magnitude: float = 0.05,
        noise_level: float = 0.005,
        sparsity_threshold: float = 0.001,
        device: str = "cpu",
    ):
        self.tensor_shape = tensor_shape
        self.random_seed = random_seed
        self.base_std = base_std
        self.task_magnitude = task_magnitude
        self.noise_level = noise_level
        self.sparsity_threshold = sparsity_threshold
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    def generate_all_blocks(self) -> List[BlockTensors]:
        """Generate all eight functional blocks with deterministic variations."""
        generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
        blocks: List[BlockTensors] = []

        # Target geometric behavior specifications per block
        block_specs = [
            # Block 0: Q1 Attention - High positive alignment
            {"align": 0.80, "conflict": 0.10, "rel_scale": 1.0, "sparse_prob": 0.05, "desc": "Aligned attention representations"},
            # Block 1: Q1 MLP - Moderate positive alignment
            {"align": 0.40, "conflict": 0.30, "rel_scale": 1.1, "sparse_prob": 0.10, "desc": "Mildly aligned early representations"},
            # Block 2: Q2 Attention - Near-orthogonal representations
            {"align": 0.00, "conflict": 0.50, "rel_scale": 0.9, "sparse_prob": 0.15, "desc": "Orthogonal attention mechanisms"},
            # Block 3: Q2 MLP - Severe sign conflict & opposing vectors
            {"align": -0.70, "conflict": 0.85, "rel_scale": 1.0, "sparse_prob": 0.10, "desc": "Strongly conflicting MLP feature neurons"},
            # Block 4: Q3 Attention - Moderate positive alignment
            {"align": 0.50, "conflict": 0.25, "rel_scale": 1.05, "sparse_prob": 0.20, "desc": "Shared mid-layer attention patterns"},
            # Block 5: Q3 MLP - Moderate negative conflict
            {"align": -0.35, "conflict": 0.65, "rel_scale": 1.2, "sparse_prob": 0.15, "desc": "Sub-network interference in MLP layers"},
            # Block 6: Q4 Attention - Highly sparse updates
            {"align": 0.20, "conflict": 0.40, "rel_scale": 0.85, "sparse_prob": 0.55, "desc": "Sparse task modifications in late attention"},
            # Block 7: Q4 MLP - High conflict with asymmetric norm ratio
            {"align": -0.55, "conflict": 0.75, "rel_scale": 1.50, "sparse_prob": 0.25, "desc": "Asymmetric divergent task vectors in late MLP"},
        ]

        for idx, (name, spec) in enumerate(zip(FUNCTIONAL_BLOCK_NAMES, block_specs)):
            block = self._generate_single_block(idx, name, spec, generator)
            blocks.append(block)

        return blocks

    def _generate_single_block(
        self,
        block_idx: int,
        name: str,
        spec: Dict[str, Any],
        generator: torch.Generator
    ) -> BlockTensors:
        """Construct synthetic tensors for a single block adhering to the given geometric specification."""
        H, W = self.tensor_shape

        # 1. Base parameter tensor (Gaussian initialization typical of pretrained LLMs)
        base = torch.randn(H, W, generator=generator, dtype=torch.float32) * self.base_std

        # 2. Shared latent direction v_shared and independent orthogonal directions
        # Coordinate-scale normalized so variance is 1.0 regardless of (H, W)
        v_shared = torch.randn(H, W, generator=generator, dtype=torch.float32)
        v_shared = v_shared / (torch.norm(v_shared, p=2) + 1e-8) * (H * W) ** 0.5

        v_ortho_math = torch.randn(H, W, generator=generator, dtype=torch.float32)
        # Gram-Schmidt to make v_ortho_math orthogonal to v_shared
        proj = (torch.sum(v_ortho_math * v_shared) / (torch.sum(v_shared * v_shared) + 1e-8)) * v_shared
        v_ortho_math = v_ortho_math - proj
        v_ortho_math = v_ortho_math / (torch.norm(v_ortho_math, p=2) + 1e-8) * (H * W) ** 0.5

        v_ortho_code = torch.randn(H, W, generator=generator, dtype=torch.float32)
        proj1 = (torch.sum(v_ortho_code * v_shared) / (torch.sum(v_shared * v_shared) + 1e-8)) * v_shared
        proj2 = (torch.sum(v_ortho_code * v_ortho_math) / (torch.sum(v_ortho_math * v_ortho_math) + 1e-8)) * v_ortho_math
        v_ortho_code = v_ortho_code - proj1 - proj2
        v_ortho_code = v_ortho_code / (torch.norm(v_ortho_code, p=2) + 1e-8) * (H * W) ** 0.5

        # 3. Construct directional components according to target alignment
        align = spec["align"]
        c_shared = abs(align) ** 0.5
        c_ortho = (max(0.0, 1.0 - abs(align))) ** 0.5

        sign_code = 1.0 if align >= 0 else -1.0
        task_math_dir = c_shared * v_shared + c_ortho * v_ortho_math
        task_code_dir = (sign_code * c_shared) * v_shared + c_ortho * v_ortho_code

        # Normalize variance back to 1.0
        task_math_dir = task_math_dir / (torch.norm(task_math_dir, p=2) + 1e-8) * (H * W) ** 0.5
        task_code_dir = task_code_dir / (torch.norm(task_code_dir, p=2) + 1e-8) * (H * W) ** 0.5

        # 4. Scale by target magnitudes (per-element standard deviation)
        mag_math = self.task_magnitude
        mag_code = self.task_magnitude * spec["rel_scale"]

        task_math = task_math_dir * mag_math
        task_code = task_code_dir * mag_code

        # 5. Add noise
        noise_math = torch.randn(H, W, generator=generator, dtype=torch.float32) * self.noise_level
        noise_code = torch.randn(H, W, generator=generator, dtype=torch.float32) * self.noise_level
        task_math = task_math + noise_math
        task_code = task_code + noise_code

        # 6. Apply sparsity mask if specified
        sparse_prob = spec.get("sparse_prob", 0.0)
        if sparse_prob > 0.0:
            mask_math = (torch.rand(H, W, generator=generator) > sparse_prob).float()
            mask_code = (torch.rand(H, W, generator=generator) > sparse_prob).float()
            task_math = task_math * mask_math
            task_code = task_code * mask_code

        metadata = {
            "is_synthetic": True,
            "disclaimer": "SYNTHETIC TENSOR DATA — NOT ACTUAL QWEN/LLM WEIGHTS",
            "block_name": name,
            "block_index": block_idx,
            "target_alignment": align,
            "target_conflict": spec["conflict"],
            "target_rel_scale": spec["rel_scale"],
            "description": spec["desc"],
            "shape": list(self.tensor_shape),
            "dtype": "torch.float32",
        }

        block = BlockTensors(
            name=name,
            block_index=block_idx,
            base=base.to(self.device),
            task_math=task_math.to(self.device),
            task_code=task_code.to(self.device),
            metadata=metadata,
        )
        return block
