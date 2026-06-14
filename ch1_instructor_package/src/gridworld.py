"""
GridWorld environment for Chapter 1 RL exercises.

The environment is intentionally lightweight and does not require Gymnasium.
It uses the modern step API idea:

    next_state, reward, terminated, truncated, info = env.step(action)

States are represented as (row, col) tuples.
Actions are integers:
    0 = up
    1 = right
    2 = down
    3 = left
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

State = Tuple[int, int]


@dataclass
class GridWorld:
    height: int = 5
    width: int = 7
    start: State = (4, 0)
    goal: State = (0, 6)
    hazards: Tuple[State, ...] = ((2, 3),)
    obstacles: Tuple[State, ...] = ((1, 1), (1, 2), (2, 1), (3, 4))
    max_steps: int = 60
    step_reward: float = -1.0
    goal_reward: float = 20.0
    hazard_reward: float = -10.0
    obstacle_reward: float = -2.0
    slip_prob: float = 0.0

    def __post_init__(self) -> None:
        self.actions = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        self.action_names = {0: "U", 1: "R", 2: "D", 3: "L"}
        self.n_actions = len(self.actions)
        self.rng = random.Random()
        self.state: State = self.start
        self.steps = 0

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)
        self.state = self.start
        self.steps = 0
        return self.state

    def states(self) -> List[State]:
        result = []
        for r in range(self.height):
            for c in range(self.width):
                s = (r, c)
                if s not in self.obstacles:
                    result.append(s)
        return result

    def action_space_sample(self) -> int:
        return self.rng.randrange(self.n_actions)

    def is_terminal(self, state: State) -> bool:
        return state == self.goal or state in self.hazards

    def _inside(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.height and 0 <= c < self.width

    def _move(self, state: State, action: int) -> Tuple[State, float]:
        dr, dc = self.actions[action]
        r, c = state
        candidate = (r + dr, c + dc)

        if not self._inside(candidate):
            return state, self.obstacle_reward
        if candidate in self.obstacles:
            return state, self.obstacle_reward
        if candidate == self.goal:
            return candidate, self.goal_reward
        if candidate in self.hazards:
            return candidate, self.hazard_reward
        return candidate, self.step_reward

    def step(self, action: int):
        if self.is_terminal(self.state):
            raise RuntimeError("Cannot call step() after termination. Call reset().")
        if action not in self.actions:
            raise ValueError(f"Invalid action {action}. Valid actions are {list(self.actions)}.")

        self.steps += 1
        current_state = self.state
        actual_action = action
        if self.slip_prob > 0.0 and self.rng.random() < self.slip_prob:
            actual_action = self.action_space_sample()

        next_state, reward = self._move(current_state, actual_action)
        self.state = next_state
        terminated = self.is_terminal(next_state)
        truncated = (self.steps >= self.max_steps) and not terminated
        info = {
            "state": current_state,
            "action": action,
            "actual_action": actual_action,
            "next_state": next_state,
        }
        return next_state, reward, terminated, truncated, info

    def render_policy(self, Q: Dict[State, np.ndarray]) -> str:
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                s = (r, c)
                if s in self.obstacles:
                    row.append("###")
                elif s == self.goal:
                    row.append(" G ")
                elif s in self.hazards:
                    row.append(" H ")
                elif s == self.start:
                    row.append(" S ")
                else:
                    if s in Q:
                        a = int(np.argmax(Q[s]))
                        row.append(f" {self.action_names[a]} ")
                    else:
                        row.append(" . ")
            lines.append("".join(row))
        return "\n".join(lines)

    def print_values(self, V: Dict[State, float]) -> None:
        for r in range(self.height):
            row = []
            for c in range(self.width):
                s = (r, c)
                if s in self.obstacles:
                    row.append("  #### ")
                elif s == self.goal:
                    row.append("   G   ")
                elif s in self.hazards:
                    row.append("   H   ")
                else:
                    row.append(f"{V.get(s, 0.0):7.2f}")
            print(" ".join(row))
