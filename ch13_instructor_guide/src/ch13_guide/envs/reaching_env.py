
"""A small Gymnasium-compatible reaching environment for Chapter 13.

The environment is intentionally lightweight. It is not a physics simulator.
Its purpose is to teach interface design, termination/truncation semantics,
reward logging, action saturation, safety filtering, and perturbation testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ReachingConfig:
    workspace_limit: float = 1.0
    max_action: float = 0.10
    success_radius: float = 0.05
    max_steps: int = 100
    action_noise_std: float = 0.0
    observation_noise_std: float = 0.0
    action_delay_steps: int = 0
    unsafe_radius: float = 0.18
    unsafe_center_x: float = 0.0
    unsafe_center_y: float = 0.0


class ReachingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Planar reaching with safety filtering and diagnostic logging.

    Observation:
        [robot_x, robot_y, goal_x, goal_y, delta_x, delta_y]

    Raw action:
        Desired Cartesian displacement in [-1, 1]^2.

    Executed action:
        Raw action scaled by `max_action`, clipped to the workspace, delayed
        if configured, and projected away from a circular unsafe region.

    Termination:
        Goal reached.

    Truncation:
        Maximum episode length reached.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: ReachingConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ReachingConfig()
        self.render_mode = render_mode
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'.")

        limit = self.config.workspace_limit
        high = np.array([limit, limit, limit, limit, 2 * limit, 2 * limit], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self._action_queue: list[np.ndarray] = []

    def _sample_point(self) -> np.ndarray:
        margin = 0.15
        limit = self.config.workspace_limit - margin
        return self.np_random.uniform(-limit, limit, size=2).astype(np.float32)

    def _observation(self) -> np.ndarray:
        noise = self.np_random.normal(
            0.0, self.config.observation_noise_std, size=4
        ).astype(np.float32)
        sensed_position = self.position + noise[:2]
        sensed_goal = self.goal + noise[2:]
        delta = sensed_goal - sensed_position
        return np.concatenate([sensed_position, sensed_goal, delta]).astype(np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _inside_unsafe_region(self, point: np.ndarray) -> bool:
        center = np.array(
            [self.config.unsafe_center_x, self.config.unsafe_center_y],
            dtype=np.float32,
        )
        return bool(np.linalg.norm(point - center) < self.config.unsafe_radius)

    def _safety_filter(self, proposed_position: np.ndarray) -> tuple[np.ndarray, bool]:
        """Project an unsafe proposal to the boundary of the unsafe circle."""
        center = np.array(
            [self.config.unsafe_center_x, self.config.unsafe_center_y],
            dtype=np.float32,
        )
        offset = proposed_position - center
        distance = float(np.linalg.norm(offset))
        if distance >= self.config.unsafe_radius:
            return proposed_position, False

        if distance < 1e-8:
            offset = np.array([1.0, 0.0], dtype=np.float32)
            distance = 1.0
        projected = center + offset / distance * self.config.unsafe_radius
        return projected.astype(np.float32), True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        self.position = np.asarray(
            options.get("initial_position", self._sample_point()),
            dtype=np.float32,
        )
        self.goal = np.asarray(
            options.get("goal", self._sample_point()),
            dtype=np.float32,
        )

        if self.position.shape != (2,) or self.goal.shape != (2,):
            raise ValueError("initial_position and goal must have shape (2,).")

        self.step_count = 0
        self._action_queue = [
            np.zeros(2, dtype=np.float32)
            for _ in range(self.config.action_delay_steps)
        ]

        info = {
            "distance": self._distance(),
            "is_success": False,
            "failure_type": None,
            "action_saturated": False,
            "safety_filter_activated": False,
            "raw_action": np.zeros(2, dtype=np.float32),
            "executed_action": np.zeros(2, dtype=np.float32),
            "reward_terms": {},
        }
        return self._observation(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), received {action.shape}.")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action must contain only finite values.")

        raw_action = action.copy()
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        action_saturated = not np.allclose(raw_action, clipped)

        scaled = clipped * self.config.max_action
        if self.config.action_noise_std > 0:
            scaled = scaled + self.np_random.normal(
                0.0, self.config.action_noise_std, size=2
            ).astype(np.float32)

        self._action_queue.append(scaled.astype(np.float32))
        executed_action = self._action_queue.pop(0)

        before = self._distance()
        proposed_position = self.position + executed_action
        limit = self.config.workspace_limit
        proposed_position = np.clip(proposed_position, -limit, limit)
        filtered_position, safety_activated = self._safety_filter(proposed_position)
        self.position = filtered_position.astype(np.float32)

        self.step_count += 1
        after = self._distance()
        progress = before - after
        action_cost = float(np.dot(executed_action, executed_action))
        safety_cost = 1.0 if safety_activated else 0.0
        success = after <= self.config.success_radius

        reward_terms = {
            "progress": progress,
            "action_cost": -0.05 * action_cost,
            "safety_cost": -0.25 * safety_cost,
            "success_bonus": 1.0 if success else 0.0,
        }
        reward = float(sum(reward_terms.values()))

        terminated = bool(success)
        truncated = bool(self.step_count >= self.config.max_steps and not terminated)
        failure_type = "timeout" if truncated else None

        info = {
            "distance": after,
            "is_success": success,
            "failure_type": failure_type,
            "action_saturated": action_saturated,
            "safety_filter_activated": safety_activated,
            "raw_action": raw_action,
            "executed_action": executed_action.copy(),
            "reward_terms": reward_terms,
            "step_count": self.step_count,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise RuntimeError("Create the environment with render_mode='rgb_array'.")
        size = 400
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)

        def world_to_pixel(point: np.ndarray) -> tuple[int, int]:
            limit = self.config.workspace_limit
            px = int((point[0] + limit) / (2 * limit) * (size - 1))
            py = int((limit - point[1]) / (2 * limit) * (size - 1))
            return px, py

        yy, xx = np.mgrid[:size, :size]
        center = np.array(
            [self.config.unsafe_center_x, self.config.unsafe_center_y],
            dtype=np.float32,
        )
        cx, cy = world_to_pixel(center)
        radius_px = int(
            self.config.unsafe_radius / (2 * self.config.workspace_limit) * size
        )
        unsafe_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
        canvas[unsafe_mask] = np.array([245, 225, 225], dtype=np.uint8)

        gx, gy = world_to_pixel(self.goal)
        rx, ry = world_to_pixel(self.position)
        for x, y, color, radius in [
            (gx, gy, np.array([40, 150, 70], dtype=np.uint8), 10),
            (rx, ry, np.array([40, 80, 190], dtype=np.uint8), 8),
        ]:
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            canvas[mask] = color

        return canvas
