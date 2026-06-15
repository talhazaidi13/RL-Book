"""Utility functions for Chapter 2 instructor-guide examples.

The functions are intentionally small and transparent. They are not complete
reinforcement-learning algorithms; they support the Chapter 2 exercise solutions
on bootstrapped targets, replay buffers, rollout buffers, target networks, and
simple algorithm-selection reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import random

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is optional for pure numpy demos
    torch = None
    F = None


def compute_q_targets(
    rewards: Sequence[float],
    dones_or_nonterminal: Sequence[float],
    next_max_q: Sequence[float],
    gamma: float,
) -> np.ndarray:
    """Compute one-step Q-learning targets.

    Args:
        rewards: Rewards r_i.
        dones_or_nonterminal: Nonterminal indicators d_i, where d_i=1 if the
            transition is nonterminal and d_i=0 if it is terminal.
        next_max_q: max_a Q_target(s_i^+, a) for each transition.
        gamma: Discount factor.

    Returns:
        y_i = r_i + gamma * d_i * next_max_q_i.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    d = np.asarray(dones_or_nonterminal, dtype=np.float64)
    next_max_q = np.asarray(next_max_q, dtype=np.float64)
    return rewards + gamma * d * next_max_q


def empirical_squared_loss(predicted_q: Sequence[float], targets: Sequence[float]) -> float:
    """Mean squared Bellman-error-style loss for selected actions."""
    predicted_q = np.asarray(predicted_q, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean((predicted_q - targets) ** 2))


def soft_target_update(theta: Sequence[float], theta_bar: Sequence[float], tau: float) -> np.ndarray:
    """Return tau * theta + (1 - tau) * theta_bar."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_bar = np.asarray(theta_bar, dtype=np.float64)
    if not (0.0 <= tau <= 1.0):
        raise ValueError("tau should lie in [0, 1].")
    return tau * theta + (1.0 - tau) * theta_bar


@dataclass
class Transition:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """A minimal replay buffer for demonstration and unit testing."""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.position = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        item = Transition(state, action, float(reward), next_state, bool(done))
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.storage):
            raise ValueError("not enough transitions to sample")
        return self.rng.sample(self.storage, batch_size)


class RolloutBuffer:
    """A simple rollout buffer for on-policy data.

    The buffer stores log probabilities at data-collection time because the
    policy may change before the loss is evaluated.
    """

    def __init__(self) -> None:
        self.states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.advantages: List[float] = []

    def add(
        self,
        state: Any,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        advantage: float = 0.0,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.advantages.append(float(advantage))

    def clear(self) -> None:
        self.__init__()

    def as_dict(self) -> Dict[str, List[Any]]:
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
        }


def dynamics_model_loss_torch(pred_next_state, true_next_state, pred_reward=None, true_reward=None):
    """Supervised dynamics/reward model loss in PyTorch.

    This is intentionally generic. For continuous states, mean squared error is
    typical; for categorical observations, a cross-entropy or negative
    log-likelihood term may be more appropriate.
    """
    if torch is None or F is None:
        raise ImportError("PyTorch is required for this function.")
    loss = F.mse_loss(pred_next_state, true_next_state)
    if pred_reward is not None and true_reward is not None:
        loss = loss + F.mse_loss(pred_reward, true_reward)
    return loss


def categorical_log_prob(logits: np.ndarray, action: int) -> float:
    """Compute log pi(a|s) for a categorical policy from logits."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return float(np.log(probs[action] + 1e-12))


def suggest_algorithm_families(
    *,
    action_space: str,
    simulator_available: bool,
    online_interaction_safe: bool,
    offline_data_available: bool,
    demonstrations_available: bool,
    safety_critical: bool,
    long_horizon: bool,
    multi_agent: bool,
) -> List[str]:
    """A rule-of-thumb algorithm family selector for discussion exercises.

    This is not a replacement for problem-specific analysis. It is a compact
    decision aid for Chapter 2 exercises.
    """
    families: List[str] = []
    action_space = action_space.lower().strip()

    if not online_interaction_safe:
        if demonstrations_available:
            families.append("imitation learning / behavior cloning as a first baseline")
        if offline_data_available:
            families.append("offline RL with conservative or behavior-constrained objectives")
        if simulator_available:
            families.append("simulation-based RL with validation before deployment")
        return families

    if action_space == "discrete":
        families.append("value-based DRL such as DQN variants")
    elif action_space == "continuous":
        families.append("actor-critic continuous-control methods such as PPO, TD3, or SAC")
    else:
        families.append("actor-critic methods with an action representation suited to the problem")

    if simulator_available:
        families.append("model-based RL or planning-assisted RL")
    if safety_critical:
        families.append("safe/constrained RL and shielded exploration")
    if long_horizon:
        families.append("hierarchical, goal-conditioned, or model-based planning methods")
    if multi_agent:
        families.append("multi-agent RL with centralized training and decentralized execution")
    if demonstrations_available:
        families.append("imitation pretraining followed by RL fine-tuning")
    return families


if __name__ == "__main__":
    q = [1.2, 0.4, -0.1]
    r = [1, 0, -1]
    d = [1, 1, 0]
    q_next = [2.0, 1.0, 0.5]
    y = compute_q_targets(r, d, q_next, gamma=0.95)
    print("targets:", y)
    print("loss:", empirical_squared_loss(q, y))
