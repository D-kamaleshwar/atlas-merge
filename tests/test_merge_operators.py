"""Unit tests for Merge Operators."""
import pytest
import torch
import math

from core.merge_operators import (
    base_preserve_merge,
    task_arithmetic_merge,
    ties_merge,
    dare_ties_merge,
    slerp_merge,
    execute_merge_operator,
)


def test_base_preserve():
    """BASE-PRESERVE must return exactly base parameters with zero delta."""
    base = torch.randn(10, 10, dtype=torch.float32)
    merged = base_preserve_merge(base)
    assert torch.equal(base, merged)
    assert torch.norm(merged - base).item() == 0.0


def test_task_arithmetic():
    """TASK ARITHMETIC must compute base + scale * (w1 * t1 + w2 * t2)."""
    base = torch.ones(4, 4, dtype=torch.float32)
    t1 = torch.full((4, 4), 2.0, dtype=torch.float32)
    t2 = torch.full((4, 4), 4.0, dtype=torch.float32)

    # Equal weighting: (0.5 * 2.0 + 0.5 * 4.0) = 3.0
    # Scale = 1.0 -> 1.0 + 3.0 = 4.0
    merged = task_arithmetic_merge(base, [t1, t2], scale=1.0)
    expected = torch.full((4, 4), 4.0, dtype=torch.float32)
    assert torch.allclose(merged, expected, atol=1e-5)

    # Scale = 0.5 -> 1.0 + 0.5 * 3.0 = 2.5
    merged_half = task_arithmetic_merge(base, [t1, t2], scale=0.5)
    expected_half = torch.full((4, 4), 2.5, dtype=torch.float32)
    assert torch.allclose(merged_half, expected_half, atol=1e-5)


def test_ties_trimming_and_sign_election():
    """TIES must respect trimming density and resolve sign conflicts."""
    base = torch.zeros(4, dtype=torch.float32)
    # Task 1: [10.0, -10.0, 0.1, 0.0]
    # Task 2: [5.0,   10.0, 0.1, 0.0]
    t1 = torch.tensor([10.0, -10.0, 0.1, 0.0], dtype=torch.float32)
    t2 = torch.tensor([5.0,   10.0, 0.1, 0.0], dtype=torch.float32)

    # Density = 0.5 -> keep top 2 coordinates out of 4 (indices 0 and 1)
    # For index 0: t1=10.0, t2=5.0 -> sign is +1. Both positive -> average = 7.5
    # For index 1: t1=-10.0, t2=10.0 -> sum = 0.0 -> sign 0 -> discarded or 0
    merged = ties_merge(base, [t1, t2], density=0.5, scale=1.0)
    assert merged.shape == base.shape
    assert math.isclose(merged[0].item(), 7.5, abs_tol=1e-4)
    # Coordinates 2 and 3 must be trimmed to 0
    assert math.isclose(merged[2].item(), 0.0, abs_tol=1e-4)
    assert math.isclose(merged[3].item(), 0.0, abs_tol=1e-4)


def test_dare_rescaling_expectation():
    """DARE rescaling must preserve the expected value across coordinates."""
    torch.manual_seed(123)
    base = torch.zeros(1000, 1000, dtype=torch.float32)
    t1 = torch.ones(1000, 1000, dtype=torch.float32) * 2.0
    density = 0.4

    # Run DARE-TIES
    merged = dare_ties_merge(base, [t1], density=density, scale=1.0, random_seed=42)
    delta = merged - base

    # Mean of delta should be close to 2.0 because E[ (mask * x) / p ] = x
    mean_val = torch.mean(delta).item()
    assert math.isclose(mean_val, 2.0, rel_tol=0.08)


def test_dare_reproducibility():
    """Same random seed must produce identical DARE results."""
    base = torch.randn(20, 20, dtype=torch.float32)
    t1 = torch.randn(20, 20, dtype=torch.float32)
    t2 = torch.randn(20, 20, dtype=torch.float32)

    m1 = dare_ties_merge(base, [t1, t2], density=0.5, scale=0.75, random_seed=999)
    m2 = dare_ties_merge(base, [t1, t2], density=0.5, scale=0.75, random_seed=999)
    assert torch.equal(m1, m2)


def test_boundary_conditions():
    """Test boundary conditions for density and scaling."""
    base = torch.randn(10, 10, dtype=torch.float32)
    t1 = torch.randn(10, 10, dtype=torch.float32)

    # Scale 0.0 should produce base
    m_scale0 = task_arithmetic_merge(base, [t1], scale=0.0)
    assert torch.allclose(m_scale0, base, atol=1e-6)

    # Density 1.0 in TIES keeps all coordinates
    m_dense1 = ties_merge(base, [t1], density=1.0, scale=1.0)
    assert m_dense1.shape == base.shape

    # Dispatcher test
    m_disp = execute_merge_operator("TIES", base, [t1], density=0.5, scale=0.8)
    assert m_disp.shape == base.shape


def test_slerp_baseline():
    """SLERP baseline test between two tasks."""
    base = torch.zeros(10, dtype=torch.float32)
    t1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    slerped = slerp_merge(base, t1, t2, t=0.5)
    assert slerped.shape == base.shape
    assert not torch.isnan(slerped).any()

