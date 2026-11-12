"""Unit tests for PPO Actor-Critic Controller."""
import pytest
import torch
import copy

from core.ppo_agent import PPOAgent, ActorCriticNetwork, PPOTransition
from utils.validation import validate_action_indices


def test_actor_critic_network_forward():
    """Actor-Critic network must output correct action logits and value shapes."""
    net = ActorCriticNetwork(
        state_dim=8,
        hidden_dim=64,
        num_operators=4,
        num_densities=4,
        num_scales=4,
    )
    s = torch.randn(5, 8, dtype=torch.float32)
    op_logits, den_logits, scale_logits, values = net(s)

    assert op_logits.shape == (5, 4)
    assert den_logits.shape == (5, 4)
    assert scale_logits.shape == (5, 4)
    assert values.shape == (5, 1)


def test_ppo_action_sampling():
    """PPO agent select_action must produce valid indices and finite values."""
    agent = PPOAgent(state_dim=8, hidden_dim=64)
    dummy_state = [0.5, 0.2, 1.1, 0.05, 0.25, 0.5, 0.5, 0.9]

    a_op, a_den, a_scale, log_prob, val = agent.select_action(dummy_state)

    assert validate_action_indices(a_op, a_den, a_scale)
    assert isinstance(log_prob, float)
    assert isinstance(val, float)
    assert not torch.isnan(torch.tensor(log_prob))
    assert not torch.isnan(torch.tensor(val))


def test_ppo_weights_update():
    """Verify that PPO parameters genuinely update after executing a policy update."""
    agent = PPOAgent(state_dim=8, hidden_dim=64, learning_rate=0.01)

    # Save copy of initial weights
    initial_params = [p.clone().detach() for p in agent.network.parameters()]

    # Collect dummy rollout transitions
    for step in range(8):
        s = [0.1 * step, 0.2, 1.0, 0.05, step / 8.0, 0.5, 0.5, 0.9]
        a_op, a_den, a_scale, lp, val = agent.select_action(s)
        trans = PPOTransition(
            state=s,
            op_action=a_op,
            density_action=a_den,
            scale_action=a_scale,
            log_prob=lp,
            value=val,
            reward=1.5,
        )
        agent.store_transition(trans)

    # Execute policy update
    log = agent.update_policy(episode=1, w_math=0.5, w_code=0.5)

    assert log.episode == 1
    assert isinstance(log.policy_loss, float)
    assert isinstance(log.value_loss, float)

    # Check that at least some network weights have changed
    has_changed = False
    for p_init, p_new in zip(initial_params, agent.network.parameters()):
        if not torch.equal(p_init, p_new):
            has_changed = True
            break

    assert has_changed, "PPO network parameters did not change during update_policy!"

