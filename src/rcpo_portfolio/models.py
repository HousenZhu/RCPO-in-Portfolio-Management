from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .config import NetworkConfig


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(input_dim: int, hidden_sizes: Iterable[int], activation_name: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_dim, hidden_size))
        layers.append(_activation(activation_name))
        current_dim = hidden_size
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: NetworkConfig) -> None:
        super().__init__()
        self.backbone = _mlp(obs_dim, config.hidden_sizes, config.activation)
        feature_dim = config.hidden_sizes[-1] if config.hidden_sizes else obs_dim
        self.policy_mean = nn.Linear(feature_dim, action_dim)
        self.reward_value = nn.Linear(feature_dim, 1)
        self.cost_value = nn.Linear(feature_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), config.init_log_std))
        self.min_log_std = config.min_log_std
        if config.equal_weight_policy_init:
            nn.init.zeros_(self.policy_mean.weight)
            nn.init.zeros_(self.policy_mean.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.policy_mean(features)
        reward_value = self.reward_value(features).squeeze(-1)
        cost_value = self.cost_value(features).squeeze(-1)
        std = torch.exp(torch.clamp(self.log_std, min=self.min_log_std)).expand_as(mean)
        return mean, std, reward_value, cost_value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, reward_value, cost_value = self.forward(obs)
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(mean, std), 1
        )
        if action is None:
            action = mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy, reward_value, cost_value

    def value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, reward_value, cost_value = self.forward(obs)
        return reward_value, cost_value
