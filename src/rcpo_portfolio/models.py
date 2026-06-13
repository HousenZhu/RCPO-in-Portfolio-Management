from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward_value: torch.Tensor
    cost_value: torch.Tensor
    branch_log_probs: torch.Tensor
    branch_entropies: torch.Tensor
    branch_reward_values: torch.Tensor
    branch_cost_values: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: NetworkConfig,
        branch_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.policy_architecture = config.policy_architecture
        self.branch_credit_mode = config.branch_credit_mode
        self.action_dim = int(action_dim)
        self.branch_sizes = [int(size) for size in (branch_sizes or [])]
        self.backbone = _mlp(obs_dim, config.hidden_sizes, config.activation)
        feature_dim = config.hidden_sizes[-1] if config.hidden_sizes else obs_dim
        self.reward_value = nn.Linear(feature_dim, 1)
        self.cost_value = nn.Linear(feature_dim, 1)
        self.min_log_std = config.min_log_std
        self.dirichlet_min_concentration = float(config.dirichlet_min_concentration)
        self.dirichlet_init_concentration = float(config.dirichlet_init_concentration)
        self.dirichlet_max_concentration = float(config.dirichlet_max_concentration)

        if self.branch_credit_mode == "standalone":
            self._validate_branch_sizes()
            self.branch_reward_values = nn.ModuleList(
                [nn.Linear(feature_dim, 1) for _ in self.branch_sizes]
            )
            self.branch_cost_values = nn.ModuleList(
                [nn.Linear(feature_dim, 1) for _ in self.branch_sizes]
            )

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
                [nn.Parameter(torch.full((size,), config.init_log_std)) for size in self.branch_sizes]
            )
            if config.equal_weight_policy_init:
                for head in self.branch_mean_heads:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
        elif self.policy_architecture in {
            "simplex_autoregressive_gaussian",
            "simplex_autoregressive_dirichlet",
        }:
            self._validate_branch_sizes()
            input_dims: list[int] = []
            previous_size = 0
            for size in self.branch_sizes:
                input_dims.append(feature_dim + previous_size)
                previous_size += size
            if self.policy_architecture == "simplex_autoregressive_gaussian":
                self.autoregressive_branch_mean_heads = nn.ModuleList(
                    [nn.Linear(input_dim, size) for input_dim, size in zip(input_dims, self.branch_sizes, strict=True)]
                )
                self.autoregressive_branch_log_stds = nn.ParameterList(
                    [nn.Parameter(torch.full((size,), config.init_log_std)) for size in self.branch_sizes]
                )
                if config.equal_weight_policy_init:
                    for head in self.autoregressive_branch_mean_heads:
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)
            else:
                self.autoregressive_dirichlet_heads = nn.ModuleList(
                    [nn.Linear(input_dim, size) for input_dim, size in zip(input_dims, self.branch_sizes, strict=True)]
                )
                probability = (
                    (self.dirichlet_init_concentration - self.dirichlet_min_concentration)
                    / (self.dirichlet_max_concentration - self.dirichlet_min_concentration)
                )
                initial_bias = math.log(probability / (1.0 - probability))
                for head in self.autoregressive_dirichlet_heads:
                    nn.init.zeros_(head.weight)
                    nn.init.constant_(head.bias, initial_bias)
        else:
            raise ValueError(f"Unsupported policy_architecture: {self.policy_architecture}")

    def _validate_branch_sizes(self) -> None:
        if not self.branch_sizes:
            raise ValueError(f"{self.policy_architecture} requires simplex branch sizes.")
        if sum(self.branch_sizes) != self.action_dim:
            raise ValueError(
                f"Branch sizes {self.branch_sizes} sum to {sum(self.branch_sizes)}, "
                f"but action_dim is {self.action_dim}."
            )
        if any(size <= 0 for size in self.branch_sizes):
            raise ValueError("Simplex branch sizes must all be positive.")

    def _values(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reward_value(features).squeeze(-1), self.cost_value(features).squeeze(-1)

    def _branch_values(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.branch_credit_mode != "standalone":
            empty = features.new_zeros((features.shape[0], 0))
            return empty, empty
        reward_values = torch.cat([head(features) for head in self.branch_reward_values], dim=-1)
        cost_values = torch.cat([head(features) for head in self.branch_cost_values], dim=-1)
        return reward_values, cost_values

    @staticmethod
    def _split_action(action: torch.Tensor, branch_sizes: Sequence[int]) -> list[torch.Tensor]:
        return list(torch.split(action, list(branch_sizes), dim=-1))

    @staticmethod
    def _branch_softmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(logits) if logits.shape[-1] == 1 else F.softmax(logits, dim=-1)

    @staticmethod
    def _stack_terms(terms: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(terms, dim=-1)

    def _fixed_branch(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = features.new_ones((features.shape[0], 1))
        zeros = features.new_zeros(features.shape[0])
        return action, zeros, zeros

    def _parallel_gaussian(
        self, features: torch.Tensor, action: torch.Tensor | None, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        provided = self._split_action(action, self.branch_sizes) if action is not None else None
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for index, (head, log_std, size) in enumerate(
            zip(self.branch_mean_heads, self.branch_log_stds, self.branch_sizes, strict=True)
        ):
            if size == 1:
                branch_action = features.new_zeros((features.shape[0], 1))
                zero = features.new_zeros(features.shape[0])
                actions.append(branch_action)
                log_probs.append(zero)
                entropies.append(zero)
                continue
            mean = head(features)
            std = torch.exp(torch.clamp(log_std, min=self.min_log_std)).expand_as(mean)
            distribution = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
            branch_action = mean if deterministic and provided is None else (
                distribution.sample() if provided is None else provided[index]
            )
            actions.append(branch_action)
            log_probs.append(distribution.log_prob(branch_action))
            entropies.append(distribution.entropy())
        return torch.cat(actions, dim=-1), self._stack_terms(log_probs), self._stack_terms(entropies)

    def _autoregressive_policy(
        self, features: torch.Tensor, action: torch.Tensor | None, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        provided = self._split_action(action, self.branch_sizes) if action is not None else None
        actions: list[torch.Tensor] = []
        previous_weights: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for index, size in enumerate(self.branch_sizes):
            branch_input = features if not previous_weights else torch.cat([features, *previous_weights], dim=-1)
            if size == 1:
                if self.policy_architecture == "simplex_autoregressive_dirichlet":
                    branch_action, log_prob, entropy = self._fixed_branch(features)
                else:
                    branch_action = features.new_zeros((features.shape[0], 1))
                    log_prob = features.new_zeros(features.shape[0])
                    entropy = features.new_zeros(features.shape[0])
                branch_weights = torch.ones_like(branch_action)
            elif self.policy_architecture == "simplex_autoregressive_dirichlet":
                raw = self.autoregressive_dirichlet_heads[index](branch_input)
                concentration = self.dirichlet_min_concentration + (
                    self.dirichlet_max_concentration - self.dirichlet_min_concentration
                ) * torch.sigmoid(raw)
                distribution = torch.distributions.Dirichlet(concentration)
                branch_action = concentration / concentration.sum(dim=-1, keepdim=True) if deterministic and provided is None else (
                    distribution.sample() if provided is None else provided[index]
                )
                branch_weights = branch_action
                log_prob = distribution.log_prob(branch_action)
                entropy = distribution.entropy()
            else:
                mean = self.autoregressive_branch_mean_heads[index](branch_input)
                std = torch.exp(
                    torch.clamp(self.autoregressive_branch_log_stds[index], min=self.min_log_std)
                ).expand_as(mean)
                distribution = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
                branch_action = mean if deterministic and provided is None else (
                    distribution.sample() if provided is None else provided[index]
                )
                branch_weights = self._branch_softmax(branch_action)
                log_prob = distribution.log_prob(branch_action)
                entropy = distribution.entropy()
            actions.append(branch_action)
            previous_weights.append(branch_weights)
            log_probs.append(log_prob)
            entropies.append(entropy)
        return torch.cat(actions, dim=-1), self._stack_terms(log_probs), self._stack_terms(entropies)

    def get_policy_output(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        features = self.backbone(obs)
        reward_value, cost_value = self._values(features)
        branch_reward_values, branch_cost_values = self._branch_values(features)
        if self.policy_architecture == "flat_gaussian":
            mean = self.policy_mean(features)
            std = torch.exp(torch.clamp(self.log_std, min=self.min_log_std)).expand_as(mean)
            distribution = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
            selected_action = mean if deterministic and action is None else (
                distribution.sample() if action is None else action
            )
            branch_log_probs = distribution.log_prob(selected_action).unsqueeze(-1)
            branch_entropies = distribution.entropy().unsqueeze(-1)
        elif self.policy_architecture == "simplex_branch_gaussian":
            selected_action, branch_log_probs, branch_entropies = self._parallel_gaussian(
                features, action, deterministic
            )
        else:
            selected_action, branch_log_probs, branch_entropies = self._autoregressive_policy(
                features, action, deterministic
            )
        return PolicyOutput(
            action=selected_action,
            log_prob=branch_log_probs.sum(dim=-1),
            entropy=branch_entropies.sum(dim=-1),
            reward_value=reward_value,
            cost_value=cost_value,
            branch_log_probs=branch_log_probs,
            branch_entropies=branch_entropies,
            branch_reward_values=branch_reward_values,
            branch_cost_values=branch_cost_values,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.get_policy_output(obs, deterministic=True)
        return output.action, torch.ones_like(output.action), output.reward_value, output.cost_value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.get_policy_output(obs, action=action, deterministic=deterministic)
        return output.action, output.log_prob, output.entropy, output.reward_value, output.cost_value

    def value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        return self._values(features)

    def value_with_branches(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        reward_value, cost_value = self._values(features)
        branch_reward_values, branch_cost_values = self._branch_values(features)
        return reward_value, cost_value, branch_reward_values, branch_cost_values
