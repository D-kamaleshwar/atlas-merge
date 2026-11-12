"""Unit tests for Pareto Archive and Hypervolume Calculation."""
import pytest
import math

from core.pareto import (
    ParetoCandidate,
    dominates,
    compute_2d_hypervolume,
    ParetoArchive,
)


def test_pareto_dominance():
    """Test standard Pareto dominance rules between two candidates."""
    cand_better = ParetoCandidate(
        candidate_id="better", episode=1, math_score=0.8, code_score=0.7,
        retention_score=0.95, reward=1.0, recipe={}
    )
    cand_worse = ParetoCandidate(
        candidate_id="worse", episode=2, math_score=0.6, code_score=0.5,
        retention_score=0.90, reward=0.5, recipe={}
    )
    cand_tradeoff = ParetoCandidate(
        candidate_id="tradeoff", episode=3, math_score=0.9, code_score=0.4,
        retention_score=0.92, reward=0.8, recipe={}
    )

    # cand_better strictly dominates cand_worse (0.8 >= 0.6 and 0.7 >= 0.5)
    assert dominates(cand_better, cand_worse)
    assert not dominates(cand_worse, cand_better)

    # cand_better and cand_tradeoff: math is better in tradeoff (0.9 > 0.8), but code is better in cand_better (0.7 > 0.4)
    # Neither should dominate the other!
    assert not dominates(cand_better, cand_tradeoff)
    assert not dominates(cand_tradeoff, cand_better)


def test_hypervolume_exact_analytical():
    """Verify hypervolume calculation against an exact manual analytical calculation.
    
    Points:
    A = (0.2, 0.8)
    B = (0.7, 0.4)
    Ref = (0.0, 0.0)
    
    Expected exact area:
    Rectangle 1: from x=0.0 to x=0.2, height=0.8 -> Area = 0.2 * 0.8 = 0.16
    Rectangle 2: from x=0.2 to x=0.7, height=0.4 -> Area = (0.7 - 0.2) * 0.4 = 0.20
    Total HV = 0.16 + 0.20 = 0.36
    """
    cand_a = ParetoCandidate(candidate_id="A", episode=1, math_score=0.2, code_score=0.8, retention_score=0.9, reward=0.0, recipe={})
    cand_b = ParetoCandidate(candidate_id="B", episode=2, math_score=0.7, code_score=0.4, retention_score=0.9, reward=0.0, recipe={})

    hv = compute_2d_hypervolume([cand_a, cand_b], reference_point=(0.0, 0.0))
    assert math.isclose(hv, 0.36, abs_tol=1e-5)


def test_pareto_archive_pruning():
    """Pareto archive should automatically prune dominated candidates upon adding superior ones."""
    archive = ParetoArchive(reference_point=(0.0, 0.0))

    c1 = ParetoCandidate(candidate_id="c1", episode=1, math_score=0.4, code_score=0.4, retention_score=0.9, reward=0.0, recipe={})
    c2 = ParetoCandidate(candidate_id="c2", episode=2, math_score=0.6, code_score=0.6, retention_score=0.9, reward=0.0, recipe={})
    c3 = ParetoCandidate(candidate_id="c3", episode=3, math_score=0.2, code_score=0.2, retention_score=0.9, reward=0.0, recipe={})

    assert archive.add_candidate(c1)
    assert len(archive.get_archive()) == 1

    # Adding c2 (dominates c1) -> c1 must be removed, c2 added
    assert archive.add_candidate(c2)
    current_arch = archive.get_archive()
    assert len(current_arch) == 1
    assert current_arch[0].candidate_id == "c2"

    # Adding c3 (dominated by c2) -> rejected, not added
    assert not archive.add_candidate(c3)
    assert len(archive.get_archive()) == 1
    assert archive.get_archive()[0].candidate_id == "c2"

