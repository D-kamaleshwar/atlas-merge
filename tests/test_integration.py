"""End-to-End Integration tests for ATLAS-MERGE."""
import pytest
import os
import shutil
import math

from core.experiment import ExperimentEngine
from core.reward import MultiObjectiveReward


def test_invalid_numerical_candidates_rejected():
    """Ensure invalid numerical candidates (NaN, Inf) receive penalty reward and are rejected."""
    reward_engine = MultiObjectiveReward(invalid_penalty=-10.0)

    # NaN in math score
    breakdown_nan = reward_engine.calculate_reward(
        math_proxy=float("nan"),
        code_proxy=0.5,
        retention_proxy=0.95,
        is_valid=True,
    )
    assert breakdown_nan.total_reward == -10.0
    assert not breakdown_nan.is_valid

    # Inf in code score
    breakdown_inf = reward_engine.calculate_reward(
        math_proxy=0.5,
        code_proxy=float("inf"),
        retention_proxy=0.95,
        is_valid=True,
    )
    assert breakdown_inf.total_reward == -10.0
    assert not breakdown_inf.is_valid


def test_retention_penalty_application():
    """Test that violating the retention floor applies the exact proportional penalty."""
    reward_engine = MultiObjectiveReward(
        rho=0.0,
        ideal_math=1.0,
        ideal_code=1.0,
        retention_floor=0.90,
        retention_penalty_weight=5.0,
    )
    # Target retention is 0.80 -> violation of 0.10 -> penalty = 5.0 * 0.10 = 0.50
    breakdown = reward_engine.calculate_reward(
        math_proxy=0.8,
        code_proxy=0.8,
        retention_proxy=0.80,
        w_math=0.5,
        w_code=0.5,
        is_valid=True,
    )
    assert breakdown.is_retention_violated
    assert math.isclose(breakdown.retention_penalty, 0.50, abs_tol=1e-5)


def test_full_experiment_pipeline(tmp_path):
    """Run a multi-episode search workflow end-to-end to verify full integration."""
    test_output_dir = str(tmp_path / "test_outputs")

    engine = ExperimentEngine(
        tensor_shape=(32, 32),
        random_seed=42,
        output_dir=test_output_dir,
    )

    assert engine.is_initialized
    assert len(engine.blocks) == 8
    assert len(engine.geometry_profiles) == 8

    # Run 3 episodes
    records = engine.run_search(num_episodes=3, w_math=0.6, w_code=0.4)

    assert len(records) == 3
    for r in records:
        assert r.math_proxy > 0.0
        assert r.code_proxy > 0.0
        assert r.retention_proxy > 0.0
        assert isinstance(r.reward, float)
        assert len(r.recipe) == 8

    # Verify export artifacts created
    reports_dir = os.path.join(test_output_dir, "reports")
    assert os.path.exists(os.path.join(reports_dir, "experiment_episodes.json"))
    assert os.path.exists(os.path.join(reports_dir, "experiment_episodes.csv"))
    assert os.path.exists(os.path.join(reports_dir, "pareto_frontier.json"))
    assert os.path.exists(os.path.join(reports_dir, "geometry_profiles.json"))

