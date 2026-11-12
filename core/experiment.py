"""Experiment Engine Orchestrator for ATLAS-MERGE.

Orchestrates the complete research simulation pipeline:
1. Synthetic task vector initialization
2. Geometric profiling
3. Geometric state assembly
4. PPO Actor-Critic action generation
5. Heterogeneous block-wise merge execution
6. Multi-objective proxy evaluation
7. Retention constraint checking
8. Pareto archive & Hypervolume updating
9. PPO policy learning updates
10. Comprehensive structured logging
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable
import os
import json
import torch

from .synthetic_data import SyntheticTaskVectorGenerator, BlockTensors, FUNCTIONAL_BLOCK_NAMES
from .geometry import profile_all_blocks, GeometryProfile, save_geometry_profiles
from .merge_operators import OPERATOR_NAMES
from .merge_engine import MergeEngine
from .evaluation import EvaluationEngine, EvaluationResult
from .reward import MultiObjectiveReward, RewardBreakdown
from .pareto import ParetoArchive, ParetoCandidate
from .ppo_agent import PPOAgent, PPOTransition, PPOTrainingLog
from utils.logging_utils import get_logger, log_jsonl
from utils.export import export_to_json, export_to_csv

logger = get_logger("atlas_merge.experiment")


@dataclass
class EpisodeRecord:
    """Full trace of an executed search episode."""
    episode: int
    candidate_id: str
    w_math: float
    w_code: float
    math_proxy: float
    code_proxy: float
    retention_proxy: float
    reward: float
    chebyshev_reward: float
    retention_penalty: float
    is_pareto_optimal: bool
    hypervolume: float
    recipe: Dict[str, Dict[str, Any]]
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentEngine:
    """End-to-end multi-objective model merge search orchestrator."""

    def __init__(
        self,
        tensor_shape: tuple = (128, 128),
        random_seed: int = 42,
        discrete_densities: Optional[List[float]] = None,
        discrete_scales: Optional[List[float]] = None,
        retention_floor: float = 0.90,
        retention_penalty_weight: float = 5.0,
        chebyshev_rho: float = 0.05,
        ppo_lr: float = 0.001,
        ppo_hidden_dim: int = 64,
        device: str = "cpu",
        output_dir: str = "outputs",
    ):
        self.tensor_shape = tensor_shape
        self.random_seed = random_seed
        self.discrete_densities = discrete_densities or [0.2, 0.4, 0.7, 1.0]
        self.discrete_scales = discrete_scales or [0.25, 0.5, 0.75, 1.0]
        self.retention_floor = retention_floor
        self.retention_penalty_weight = retention_penalty_weight
        self.chebyshev_rho = chebyshev_rho
        self.ppo_lr = ppo_lr
        self.ppo_hidden_dim = ppo_hidden_dim
        self.device = device
        self.output_dir = output_dir

        self.blocks: List[BlockTensors] = []
        self.geometry_profiles: Dict[str, GeometryProfile] = {}
        self.merge_engine = MergeEngine(
            discrete_densities=self.discrete_densities,
            discrete_scales=self.discrete_scales,
        )
        self.eval_engine = EvaluationEngine()
        self.reward_engine = MultiObjectiveReward(
            rho=self.chebyshev_rho,
            retention_floor=self.retention_floor,
            retention_penalty_weight=self.retention_penalty_weight,
        )
        self.pareto_archive = ParetoArchive(reference_point=(0.0, 0.0))
        self.ppo_agent = PPOAgent(
            state_dim=8,
            hidden_dim=self.ppo_hidden_dim,
            num_operators=len(OPERATOR_NAMES),
            num_densities=len(self.discrete_densities),
            num_scales=len(self.discrete_scales),
            learning_rate=self.ppo_lr,
            device=self.device,
        )

        self.episode_records: List[EpisodeRecord] = []
        self.is_initialized: bool = False
        self.initialize()
        self.load_existing_results()

    def initialize(self) -> None:
        """Initialize synthetic vectors, profile geometry, and prepare state."""
        torch.manual_seed(self.random_seed)

        generator = SyntheticTaskVectorGenerator(
            tensor_shape=self.tensor_shape,
            random_seed=self.random_seed,
            device=self.device,
        )
        self.blocks = generator.generate_all_blocks()
        self.geometry_profiles = profile_all_blocks(self.blocks)
        self.is_initialized = True

    def load_existing_results(self) -> None:
        """Load previously saved experiment runs from disk if present."""
        reports_dir = os.path.join(self.output_dir, "reports")
        episodes_path = os.path.join(reports_dir, "experiment_episodes.json")
        if os.path.exists(episodes_path):
            try:
                with open(episodes_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.episode_records = [EpisodeRecord(**d) for d in data]
                # Reconstruct pareto archive
                self.pareto_archive.clear()
                for rec in self.episode_records:
                    cand = ParetoCandidate(
                        candidate_id=rec.candidate_id,
                        episode=rec.episode,
                        math_score=rec.math_proxy,
                        code_score=rec.code_proxy,
                        retention_score=rec.retention_proxy,
                        reward=rec.reward,
                        recipe=rec.recipe,
                    )
                    self.pareto_archive.add_candidate(cand)
            except Exception as e:
                logger.warning(f"Could not load existing experiment results: {e}")

    def reset(self, new_seed: Optional[int] = None) -> None:
        """Reset the experiment state, reseed, and clear archives."""
        if new_seed is not None:
            self.random_seed = new_seed
        self.pareto_archive.clear()
        self.episode_records.clear()
        self.ppo_agent = PPOAgent(
            state_dim=8,
            hidden_dim=self.ppo_hidden_dim,
            num_operators=len(OPERATOR_NAMES),
            num_densities=len(self.discrete_densities),
            num_scales=len(self.discrete_scales),
            learning_rate=self.ppo_lr,
            device=self.device,
        )
        self.initialize()

    def run_search(
        self,
        num_episodes: int = 25,
        w_math: float = 0.5,
        w_code: float = 0.5,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> List[EpisodeRecord]:
        """Execute multi-objective merge search over a budget of episodes."""
        if not self.is_initialized:
            self.initialize()

        log_jsonl_path = os.path.join(self.output_dir, "checkpoints", "ppo_training_logs.jsonl")

        for ep in range(1, num_episodes + 1):
            cand_id = f"cand_{ep:03d}"
            recipe: Dict[str, Dict[str, Any]] = {}
            transitions_in_episode: List[PPOTransition] = []

            # 1. Block-wise state generation and action sampling
            for b_idx, block in enumerate(self.blocks):
                geom = self.geometry_profiles[block.name]
                # Construct 8-dimensional state vector
                depth_norm = float(b_idx / max(1, len(self.blocks) - 1))
                budget_remaining = float(1.0 - (ep / num_episodes))
                state = [
                    float(geom.cosine_similarity),
                    float(geom.sign_conflict_density),
                    float(geom.relative_norm),
                    float(geom.sparsity_mean),
                    depth_norm,
                    float(w_math),
                    float(w_code),
                    budget_remaining,
                ]

                a_op, a_den, a_scale, log_prob, val = self.ppo_agent.select_action(state)

                op_name = OPERATOR_NAMES[a_op]
                den_val = self.discrete_densities[a_den]
                scale_val = self.discrete_scales[a_scale]

                recipe[block.name] = {
                    "operator": op_name,
                    "density": den_val,
                    "scale": scale_val,
                    "op_idx": a_op,
                    "density_idx": a_den,
                    "scale_idx": a_scale,
                }

                transition = PPOTransition(
                    state=state,
                    op_action=a_op,
                    density_action=a_den,
                    scale_action=a_scale,
                    log_prob=log_prob,
                    value=val,
                )
                transitions_in_episode.append(transition)

            # 2. Execute block-wise merging
            merged_tensors = self.merge_engine.merge_blocks(
                blocks=self.blocks,
                recipe=recipe,
                random_seed=self.random_seed + ep,
            )

            # 3. Evaluate multi-objective synthetic proxies
            eval_result: EvaluationResult = self.eval_engine.evaluate_merged_model(
                blocks=self.blocks,
                merged_tensors=merged_tensors,
                geometry_profiles=self.geometry_profiles,
                recipes=recipe,
            )

            # 4. Compute preference-conditioned Augmented Chebyshev reward
            reward_info: RewardBreakdown = self.reward_engine.calculate_reward(
                math_proxy=eval_result.math_proxy,
                code_proxy=eval_result.code_proxy,
                retention_proxy=eval_result.retention_proxy,
                w_math=w_math,
                w_code=w_code,
                is_valid=eval_result.is_valid,
            )

            # 5. Assign reward to rollout transitions and store in agent buffer
            for trans in transitions_in_episode:
                trans.reward = reward_info.total_reward
                self.ppo_agent.store_transition(trans)

            # 6. Update Pareto Archive
            candidate = ParetoCandidate(
                candidate_id=cand_id,
                episode=ep,
                math_score=eval_result.math_proxy,
                code_score=eval_result.code_proxy,
                retention_score=eval_result.retention_proxy,
                reward=reward_info.total_reward,
                recipe=recipe,
            )
            is_pareto = self.pareto_archive.add_candidate(candidate)
            current_hv = self.pareto_archive.get_current_hypervolume()

            # 7. Execute PPO Policy Update
            p_log: PPOTrainingLog = self.ppo_agent.update_policy(
                episode=ep,
                w_math=w_math,
                w_code=w_code,
                log_filepath=log_jsonl_path,
            )

            # 8. Record episode summary
            record = EpisodeRecord(
                episode=ep,
                candidate_id=cand_id,
                w_math=w_math,
                w_code=w_code,
                math_proxy=eval_result.math_proxy,
                code_proxy=eval_result.code_proxy,
                retention_proxy=eval_result.retention_proxy,
                reward=reward_info.total_reward,
                chebyshev_reward=reward_info.chebyshev_reward,
                retention_penalty=reward_info.retention_penalty,
                is_pareto_optimal=is_pareto,
                hypervolume=current_hv,
                recipe=recipe,
                policy_loss=p_log.policy_loss,
                value_loss=p_log.value_loss,
                entropy=p_log.entropy,
            )
            self.episode_records.append(record)

            if progress_callback:
                progress_callback(ep, num_episodes, record.to_dict())

        # Save experiment logs and archive
        self._export_results()
        return self.episode_records

    def _export_results(self) -> None:
        """Export experiment logs and Pareto frontier to JSON and CSV."""
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # 1. Episode logs
        records_dict = [r.to_dict() for r in self.episode_records]
        export_to_json(records_dict, os.path.join(reports_dir, "experiment_episodes.json"))
        export_to_csv(records_dict, os.path.join(reports_dir, "experiment_episodes.csv"))

        # 2. Pareto front
        pareto_dict = [c.to_dict() for c in self.pareto_archive.get_archive()]
        export_to_json(pareto_dict, os.path.join(reports_dir, "pareto_frontier.json"))
        export_to_csv(pareto_dict, os.path.join(reports_dir, "pareto_frontier.csv"))

        # 3. Geometry profiles
        save_geometry_profiles(self.geometry_profiles, os.path.join(reports_dir, "geometry_profiles.json"))
