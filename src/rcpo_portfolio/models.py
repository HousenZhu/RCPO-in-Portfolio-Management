from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

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
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: NetworkConfig,
        branch_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.policy_architecture = (
            "simplex_autoregressive_gaussian"
            if config.policy_architecture == "simplex_autoregressive_dirichlet"
            else config.policy_architecture
        )
        self.action_dim = int(action_dim)
        self.branch_sizes = [int(size) for size in (branch_sizes or [])]
        self.backbone = _mlp(obs_dim, config.hidden_sizes, config.activation)
        feature_dim = config.hidden_sizes[-1] if config.hidden_sizes else obs_dim
        self.reward_value = nn.Linear(feature_dim, 1)
        self.cost_value = nn.Linear(feature_dim, 1)
        self.min_log_std = config.min_log_std

        if self.policy_architecture == "flat_gaussian":
            self.policy_mean = nn.Linear(feature_dim, self.action_dim)
            self.log_std = nn.Parameter(torch.full((self.action_dim,), config.init_log_std))
            if config.equal_weight_policy_init:
                nn.init.zeros_(self.policy_mean.weight)
                nn.init.zeros_(self.policy_mean.bias)
        elif self.policy_architecture == "simplex_branch_gaussian":
            self._validate_branch_sizes()
            self.branch_mean_heads = nn.ModuleList(
                [nn.Linear(feature_dim, size) for size in self.branch_sizes]
            )
            self.branch_log_stds = nn.ParameterList(
                [
                    nn.Parameter(torch.full((size,), config.init_log_std))
                    for size in self.branch_sizes
                ]
            )
            if config.equal_weight_policy_init:
                for head in self.branch_mean_heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
        elif self.policy_architecture == "simplex_autoregressive_gaussian":
            self._validate_branch_sizes()
            autoregressive_input_dims: list[int] = []
            previous_size = 0
            for size in self.branch_sizes:
                autoregressive_input_dims.append(feature_dim + previous_size)
                previous_size += size
            self.autoregressive_branch_mean_heads = nn.ModuleList(
                [
                    nn.Linear(input_dim, size)
                    for input_dim, size in zip(
                        autoregressive_input_dims,
                        self.branch_sizes,
                        strict=True,
                    )
                ]
            )
            self.autoregressive_branch_log_stds = nn.ParameterList(
                [
                    nn.Parameter(torch.full((size,), config.init_log_std))
                    for size in self.branch_sizes
                ]
            )
            if config.equal_weight_policy_init:
                for head in self.autoregressive_branch_mean_heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
        else:
            raise ValueError(f"Unsupported policy_architecture: {self.policy_architecture}")

    def _validate_branch_sizes(self) -> None:
        if not self.branch_sizes:
            raise ValueError(
                f"{self.policy_architecture} requires simplex branch sizes."
            )
        if sum(self.branch_sizes) != self.action_dim:
            raise ValueError(
                f"Branch sizes {self.branch_sizes} sum to {sum(self.branch_sizes)}, "
                f"but action_dim is {self.action_dim}."
            )
        if any(size <= 0 for size in self.branch_sizes):
            raise ValueError("Simplex branch sizes must all be positive.")

    def _values(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reward_value = self.reward_value(features).squeeze(-1)
        cost_value = self.cost_value(features).squeeze(-1)
        return reward_value, cost_value

    def _gaussian_params(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.policy_architecture == "flat_gaussian":
            mean = self.policy_mean(features)
            std = torch.exp(torch.clamp(self.log_std, min=self.min_log_std)).expand_as(mean)
            return mean, std
        means = [head(features) for head in self.branch_mean_heads]
        stds = [
            torch.exp(torch.clamp(log_std, min=self.min_log_std)).expand_as(mean)
            for log_std, mean in zip(self.branch_log_stds, means, strict=True)
        ]
        return torch.cat(means, dim=-1), torch.cat(stds, dim=-1)

    @staticmethod
    def _split_action(action: torch.Tensor, branch_sizes: Sequence[int]) -> list[torch.Tensor]:
        branches: list[torch.Tensor] = []
        offset = 0
        for size in branch_sizes:
            branches.append(action[..., offset : offset + size])
            offset += size
        return branches

    @staticmethod
    def _branch_softmax(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return torch.ones_like(logits)
        return F.softmax(logits, dim=-1)

    def _autoregressive_gaussian_action_log_prob_entropy(
        self,
        features: torch.Tensor,
        action: torch.Tensor | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        provided_branches = (
            self._split_action(action, self.branch_sizes) if action is not None else None
        )
        branch_logits: list[torch.Tensor] = []
        previous_branch_weights: list[torch.Tensor] = []
        log_prob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        batch_size = features.shape[0]
        device = features.device
        dtype = features.dtype
        for branch_index, branch_size in enumerate(self.branch_sizes):
            branch_input = (
                features
                if not previous_branch_weights
                else torch.cat([features, *previous_branch_weights], dim=-1)
            )
            if branch_size == 1:
                branch_action = torch.zeros((batch_size, 1), device=device, dtype=dtype)
                branch_logits.append(branch_action)
                previous_branch_weights.append(torch.ones_like(branch_action))
                log_prob_terms.append(torch.zeros(batch_size, device=device, dtype=dtype))
                entropy_terms.append(torch.zeros(batch_size, device=device, dtype=dtype))
                continue
            mean = self.autoregressive_branch_mean_heads[branch_index](branch_input)
            std = torch.exp(
                torch.clamp(
                    self.autoregressive_branch_log_stds[branch_index],
                    min=self.min_log_std,
                )
            ).expand_as(mean)
            distribution = torch.distributions.Independent(
                torch.distributions.Normal(mean, std),
                1,
            )
            if provided_branches is None:
                branch_action = mean if deterministic else distribution.sample()
            else:
                branch_action = provided_branches[branch_index]
            branch_logits.append(branch_action)
            previous_branch_weights.append(self._branch_softmax(branch_action))
            log_prob_terms.append(distribution.log_prob(branch_action))
            entropy_terms.append(distribution.entropy())
        return (
            torch.cat(branch_logits, dim=-1),
            torch.stack(log_prob_terms, dim=0).sum(dim=0),
            torch.stack(entropy_terms, dim=0).sum(dim=0),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        reward_value, cost_value = self._values(features)
        if self.policy_architecture == "simplex_autoregressive_gaussian":
            action, _, _ = self._autoregressive_gaussian_action_log_prob_entropy(
                features,
                action=None,
                deterministic=True,
            )
            return action, torch.ones_like(action), reward_value, cost_value
        mean, std = self._gaussian_params(features)
        return mean, std, reward_value, cost_value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        reward_value, cost_value = self._values(features)
        if self.policy_architecture == "simplex_autoregressive_gaussian":
            action, log_prob, entropy = self._autoregressive_gaussian_action_log_prob_entropy(
                features,
                action=action,
                deterministic=deterministic,
            )
            return action, log_prob, entropy, reward_value, cost_value

        mean, std = self._gaussian_params(features)
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            1,
        )
        if action is None:
            action = mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy, reward_value, cost_value

    def value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        return self._values(features)
