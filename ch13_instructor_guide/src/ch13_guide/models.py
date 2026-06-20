
"""Small PyTorch models used in the instructor notebooks."""

from __future__ import annotations

import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, observation_dim: int = 6, action_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class ConditionalDenoiser(nn.Module):
    """Minimal denoiser for action-chunk demonstrations."""

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        total_action_dim = action_horizon * action_dim
        self.network = nn.Sequential(
            nn.Linear(total_action_dim + condition_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, total_action_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        diffusion_level: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_actions.shape[0]
        flattened = noisy_actions.reshape(batch, -1)
        level = diffusion_level.reshape(batch, 1).to(flattened.dtype)
        output = self.network(torch.cat([flattened, condition, level], dim=-1))
        return output.reshape(batch, self.action_horizon, self.action_dim)
