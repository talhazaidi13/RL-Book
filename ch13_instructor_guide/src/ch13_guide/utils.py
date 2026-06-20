
"""Utility functions for reproducible Chapter 13 experiments."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def proportional_controller(observation: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """A scripted reaching demonstrator using the observed goal displacement."""
    delta = observation[-2:]
    return np.clip(gain * delta, -1.0, 1.0).astype(np.float32)


def rollout(
    env,
    policy: Callable[[np.ndarray], np.ndarray],
    episodes: int = 20,
    seed: int = 0,
) -> list[dict]:
    records: list[dict] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        safety_interventions = 0
        saturations = 0
        while True:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            safety_interventions += int(info["safety_filter_activated"])
            saturations += int(info["action_saturated"])
            if terminated or truncated:
                records.append(
                    {
                        "episode": episode,
                        "return": total_reward,
                        "success": bool(info["is_success"]),
                        "steps": int(info["step_count"]),
                        "final_distance": float(info["distance"]),
                        "safety_interventions": safety_interventions,
                        "action_saturations": saturations,
                        "failure_type": info["failure_type"],
                    }
                )
                break
    return records
