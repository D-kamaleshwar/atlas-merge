"""Unit tests for Task Vector Geometry Profiler."""
import pytest
import torch
import math

from core.geometry import (
    compute_cosine_similarity,
    compute_sign_conflict_density,
    compute_relative_l2_norm,
    compute_sparsity,
    profile_single_block,
    profile_all_blocks,
    GeometryProfile,
)
from core.synthetic_data import BlockTensors, SyntheticTaskVectorGenerator


def test_cosine_similarity_identical_vectors():
    """Identical non-zero vectors should have cosine similarity approximately 1.0."""
    v = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    sim = compute_cosine_similarity(v, v)
    assert math.isclose(sim, 1.0, abs_tol=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors should have cosine similarity approximately 0.0."""
    v1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    v2 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    sim = compute_cosine_similarity(v1, v2)
    assert math.isclose(sim, 0.0, abs_tol=1e-5)


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors should have cosine similarity approximately -1.0."""
    v1 = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    v2 = -v1
    sim = compute_cosine_similarity(v1, v2)
    assert math.isclose(sim, -1.0, abs_tol=1e-5)


def test_sign_conflict_density_opposites():
    """Vectors with strictly opposite active signs must yield sign conflict density 1.0."""
    v1 = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
    v2 = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float32)
    conflict = compute_sign_conflict_density(v1, v2, magnitude_threshold=1e-4)
    assert math.isclose(conflict, 1.0, abs_tol=1e-5)


def test_sign_conflict_density_identical():
    """Vectors with identical signs must yield sign conflict density 0.0."""
    v1 = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
    v2 = torch.tensor([2.0, -1.0, 0.5, -3.0], dtype=torch.float32)
    conflict = compute_sign_conflict_density(v1, v2, magnitude_threshold=1e-4)
    assert math.isclose(conflict, 0.0, abs_tol=1e-5)


def test_relative_l2_norm():
    """Relative norm norm(B)/norm(A) should be mathematically exact."""
    v1 = torch.tensor([3.0, 4.0], dtype=torch.float32)   # norm = 5.0
    v2 = torch.tensor([6.0, 8.0], dtype=torch.float32)   # norm = 10.0
    rel_norm = compute_relative_l2_norm(v1, v2)
    assert math.isclose(rel_norm, 2.0, rel_tol=1e-4)


def test_sparsity():
    """Sparsity should reflect fraction of coordinates below threshold."""
    # 4 elements: 2 are <= 0.001 (0.0001, 0.0), 2 are > 0.001 (0.5, 1.0)
    t = torch.tensor([0.0001, 0.0, 0.5, 1.0], dtype=torch.float32)
    sp = compute_sparsity(t, threshold=0.001)
    assert math.isclose(sp, 0.5, abs_tol=1e-5)


def test_nan_inf_protection():
    """Geometry calculation should handle NaN/Inf gracefully without crashing."""
    v_nan = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    v_clean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    cos_sim = compute_cosine_similarity(v_nan, v_clean)
    assert cos_sim == 0.0

    conflict = compute_sign_conflict_density(v_nan, v_clean)
    assert conflict == 0.0


def test_profile_all_eight_blocks():
    """Verify that all eight blocks produce structured profiles with expected variations."""
    generator = SyntheticTaskVectorGenerator(tensor_shape=(64, 64), random_seed=42)
    blocks = generator.generate_all_blocks()
    assert len(blocks) == 8

    profiles = profile_all_blocks(blocks)
    assert len(profiles) == 8

    # Verify Q1 Attention has positive alignment
    assert profiles["Q1 Attention"].cosine_similarity > 0.4
    # Verify Q2 MLP has severe negative conflict
    assert profiles["Q2 MLP"].cosine_similarity < -0.3
    assert profiles["Q2 MLP"].sign_conflict_density > 0.6
    # Verify profiles are valid
    for p in profiles.values():
        assert p.is_valid
        assert -1.0 <= p.cosine_similarity <= 1.0
        assert 0.0 <= p.sign_conflict_density <= 1.0

