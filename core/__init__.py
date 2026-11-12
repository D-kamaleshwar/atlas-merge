"""ATLAS-MERGE Core Modules."""
from .synthetic_data import SyntheticTaskVectorGenerator, BlockTensors, FUNCTIONAL_BLOCK_NAMES
from .geometry import (
    GeometryProfile,
    compute_cosine_similarity,
    compute_sign_conflict_density,
    compute_relative_l2_norm,
    compute_sparsity,
    profile_single_block,
    profile_all_blocks,
    save_geometry_profiles,
    load_geometry_profiles,
)
from .merge_operators import (
    OPERATOR_NAMES,
    base_preserve_merge,
    task_arithmetic_merge,
    ties_merge,
    dare_ties_merge,
    slerp_merge,
    execute_merge_operator,
)
from .merge_engine import MergeEngine
from .evaluation import EvaluationEngine, EvaluationResult
from .reward import MultiObjectiveReward, RewardBreakdown
from .pareto import ParetoArchive, ParetoCandidate, dominates, compute_2d_hypervolume
from .ppo_agent import PPOAgent, PPOTransition, PPOTrainingLog, ActorCriticNetwork
from .experiment import ExperimentEngine, EpisodeRecord

__all__ = [
    "SyntheticTaskVectorGenerator",
    "BlockTensors",
    "FUNCTIONAL_BLOCK_NAMES",
    "GeometryProfile",
    "compute_cosine_similarity",
    "compute_sign_conflict_density",
    "compute_relative_l2_norm",
    "compute_sparsity",
    "profile_single_block",
    "profile_all_blocks",
    "save_geometry_profiles",
    "load_geometry_profiles",
    "OPERATOR_NAMES",
    "base_preserve_merge",
    "task_arithmetic_merge",
    "ties_merge",
    "dare_ties_merge",
    "slerp_merge",
    "execute_merge_operator",
    "MergeEngine",
    "EvaluationEngine",
    "EvaluationResult",
    "MultiObjectiveReward",
    "RewardBreakdown",
    "ParetoArchive",
    "ParetoCandidate",
    "dominates",
    "compute_2d_hypervolume",
    "PPOAgent",
    "PPOTransition",
    "PPOTrainingLog",
    "ActorCriticNetwork",
    "ExperimentEngine",
    "EpisodeRecord",
]

