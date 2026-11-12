"""PPO Actor-Critic Controller for ATLAS-MERGE.

Implements an authentic, trainable Proximal Policy Optimization (PPO) reinforcement learning agent
to search for optimal block-wise model merge recipes.
"""
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils.logging_utils import log_jsonl
from utils.validation import check_numerical_validity


@dataclass
class PPOTransition:
    """Stores experience transition for a single block decision."""
    state: List[float]
    op_action: int
    density_action: int
    scale_action: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


@dataclass
class PPOTrainingLog:
    """Structured log record for a PPO update step."""
    episode: int
    policy_loss: float
    value_loss: float
    entropy: float
    mean_reward: float
    total_loss: float
    math_preference: float
    code_preference: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActorCriticNetwork(nn.Module):
    """Shared two-layer MLP encoder with independent categorical action heads and critic value head."""

    def __init__(
        self,
        state_dim: int = 8,
        hidden_dim: int = 64,
        num_operators: int = 4,
        num_densities: int = 4,
        num_scales: int = 4,
    ):
        super().__init__()
        # Shared 2-layer encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Categorical Policy Heads
        self.operator_head = nn.Linear(hidden_dim, num_operators)
        self.density_head = nn.Linear(hidden_dim, num_densities)
        self.scale_head = nn.Linear(hidden_dim, num_scales)

        # Critic Value Head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action logits and scalar value estimate from state tensor."""
        features = self.shared_encoder(state)
        op_logits = self.operator_head(features)
        density_logits = self.density_head(features)
        scale_logits = self.scale_head(features)
        value = self.value_head(features)
        return op_logits, density_logits, scale_logits, value


class PPOAgent:
    """PPO Controller that learns block-wise operator, density, and scale selections."""

    def __init__(
        self,
        state_dim: int = 8,
        hidden_dim: int = 64,
        num_operators: int = 4,
        num_densities: int = 4,
        num_scales: int = 4,
        gamma: float = 1.0,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_operators = num_operators
        self.num_densities = num_densities
        self.num_scales = num_scales
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        # Initialize Actor-Critic network and Adam optimizer
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_operators=num_operators,
            num_densities=num_densities,
            num_scales=num_scales,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.rollout_buffer: List[PPOTransition] = []
        self.training_logs: List[PPOTrainingLog] = []

    def select_action(self, state_list: List[float]) -> Tuple[int, int, int, float, float]:
        """Given an 8-dim state vector, sample categorical actions and return log prob and value."""
        self.network.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(state_list, dtype=torch.float32, device=self.device).unsqueeze(0)
            op_logits, density_logits, scale_logits, value_tensor = self.network(s_tensor)

            dist_op = Categorical(logits=op_logits)
            dist_den = Categorical(logits=density_logits)
            dist_scale = Categorical(logits=scale_logits)

            a_op = dist_op.sample()
            a_den = dist_den.sample()
            a_scale = dist_scale.sample()

            log_prob = (
                dist_op.log_prob(a_op)
                + dist_den.log_prob(a_den)
                + dist_scale.log_prob(a_scale)
            ).item()

            value = value_tensor.squeeze().item()

        return a_op.item(), a_den.item(), a_scale.item(), float(log_prob), float(value)

    def store_transition(self, transition: PPOTransition) -> None:
        """Add transition to buffer."""
        self.rollout_buffer.append(transition)

    def update_policy(
        self,
        episode: int,
        w_math: float,
        w_code: float,
        log_filepath: Optional[str] = None,
    ) -> PPOTrainingLog:
        """Execute authentic PPO policy update using collected rollout transitions."""
        if not self.rollout_buffer:
            return PPOTrainingLog(
                episode=episode,
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                mean_reward=0.0,
                total_loss=0.0,
                math_preference=w_math,
                code_preference=w_code,
            )

        self.network.train()

        states = torch.tensor([t.state for t in self.rollout_buffer], dtype=torch.float32, device=self.device)
        actions_op = torch.tensor([t.op_action for t in self.rollout_buffer], dtype=torch.long, device=self.device)
        actions_den = torch.tensor([t.density_action for t in self.rollout_buffer], dtype=torch.long, device=self.device)
        actions_scale = torch.tensor([t.scale_action for t in self.rollout_buffer], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.rollout_buffer], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in self.rollout_buffer], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t.value for t in self.rollout_buffer], dtype=torch.float32, device=self.device)

        # Advantage calculation with gamma = 1.0
        # Returns = rewards
        returns = rewards
        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass on updated network
        op_logits, density_logits, scale_logits, new_values = self.network(states)
        new_values = new_values.squeeze(-1)

        dist_op = Categorical(logits=op_logits)
        dist_den = Categorical(logits=density_logits)
        dist_scale = Categorical(logits=scale_logits)

        new_log_probs = (
            dist_op.log_prob(actions_op)
            + dist_den.log_prob(actions_den)
            + dist_scale.log_prob(actions_scale)
        )

        entropy = (dist_op.entropy() + dist_den.entropy() + dist_scale.entropy()).mean()

        # Ratio and Clipped Surrogate Loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Loss
        value_loss = 0.5 * nn.functional.mse_loss(new_values, returns)

        # Total Loss
        total_loss = policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)

        # Parameter gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        log_record = PPOTrainingLog(
            episode=episode,
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy.item()),
            mean_reward=float(rewards.mean().item()),
            total_loss=float(total_loss.item()),
            math_preference=w_math,
            code_preference=w_code,
        )

        self.training_logs.append(log_record)

        # Clear buffer for next episode / rollout
        self.rollout_buffer = []

        if log_filepath:
            log_jsonl(log_record.to_dict(), log_filepath)

        return log_record

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights and optimizer state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model weights and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

