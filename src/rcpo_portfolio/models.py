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
        branch_train_mask: Sequence[bool] | None = None,
    ) -> None:
        super().__init__()
        self.policy_architecture = config.policy_architecture
        self.branch_credit_mode = config.branch_credit_mode
        self.action_dim = int(action_dim)
        self.branch_sizes = [int(size) for size in (branch_sizes or [])]
        self.branch_train_mask = (
            [bool(active) for active in branch_train_mask]
            if branch_train_mask is not None
            else [True for _ in self.branch_sizes]
        )
        self.backbone = _mlp(obs_dim, config.hidden_sizes, config.activation)
        feature_dim = config.hidden_sizes[-1] if config.hidden_sizes else obs_dim
        self.reward_value = nn.Linear(feature_dim, 1)
        self.cost_value = nn.Linear(feature_dim, 1)
        self.min_log_std = config.min_log_std
        self.dirichlet_min_concentration = float(config.dirichlet_min_concentration)
        self.dirichlet_init_concentration = float(config.dirichlet_init_concentration)
        self.dirichlet_max_concentration = float(config.dirichlet_max_concentration)

        if self.branch_credit_mode != "global":
            self._validate_branch_sizes()
            self.branch_reward_values = nn.ModuleList(
                [nn.Linear(feature_dim, 1) for _ in self.branch_sizes]
            )
            if self.branch_credit_mode == "standalone":
                self.branch_cost_values = nn.ModuleList(
                    [nn.Linear(feature_dim, 1) for _ in self.branch_sizes]
                )

        if self.policy_architecture == "flat_gaussian":
            self.policy_mean = nn.Linear(feature_dim, self.action_dim)
            self.log_std = nn.Parameter(torch.full((self.action_dim,), config.init_log_std))
            if self.branch_sizes:
                self._validate_branch_sizes()
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
        if len(self.branch_train_mask) != len(self.branch_sizes):
            raise ValueError(
                "branch_train_mask must have one entry per simplex branch."
            )

    def _values(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reward_value(features).squeeze(-1), self.cost_value(features).squeeze(-1)

    def _branch_values(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.branch_credit_mode == "global":
            empty = features.new_zeros((features.shape[0], 0))
            return empty, empty
        reward_values = torch.cat(
            [
                head(features) if active else features.new_zeros((features.shape[0], 1))
                for head, active in zip(
                    self.branch_reward_values, self.branch_train_mask, strict=True
                )
            ],
            dim=-1,
        )
        if self.branch_credit_mode == "standalone":
            cost_values = torch.cat(
                [
                    head(features) if active else features.new_zeros((features.shape[0], 1))
                    for head, active in zip(
                        self.branch_cost_values, self.branch_train_mask, strict=True
                    )
                ],
                dim=-1,
            )
        else:
            cost_values = features.new_zeros(
                (features.shape[0], len(self.branch_sizes))
            )
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
            if not self.branch_train_mask[index]:
                branch_action = features.new_zeros((features.shape[0], size))
                zero = features.new_zeros(features.shape[0])
                actions.append(branch_action)
                log_probs.append(zero)
                entropies.append(zero)
                continue
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
            if not self.branch_train_mask[index]:
                if self.policy_architecture == "simplex_autoregressive_dirichlet":
                    branch_action = features.new_full(
                        (features.shape[0], size), 1.0 / float(size)
                    )
                else:
                    branch_action = features.new_zeros((features.shape[0], size))
                branch_weights = features.new_full(
                    (features.shape[0], size), 1.0 / float(size)
                )
                log_prob = features.new_zeros(features.shape[0])
                entropy = features.new_zeros(features.shape[0])
                actions.append(branch_action)
                previous_weights.append(branch_weights)
                log_probs.append(log_prob)
                entropies.append(entropy)
                continue
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
            component_distribution = torch.distributions.Normal(mean, std)
            selected_action = mean if deterministic and action is None else (
                component_distribution.sample() if action is None else action
            )
            if self.branch_sizes:
                active_components = torch.as_tensor(
                    [
                        active
                        for size, active in zip(
                            self.branch_sizes, self.branch_train_mask, strict=True
                        )
                        for _ in range(size)
                    ],
                    dtype=torch.bool,
                    device=features.device,
                )
                selected_action = torch.where(
                    active_components.unsqueeze(0),
                    selected_action,
                    torch.zeros_like(selected_action),
                )
                active_float = active_components.to(dtype=features.dtype).unsqueeze(0)
                branch_log_probs = (
                    component_distribution.log_prob(selected_action) * active_float
                ).sum(dim=-1, keepdim=True)
                branch_entropies = (
                    component_distribution.entropy() * active_float
                ).sum(dim=-1, keepdim=True)
            else:
                branch_log_probs = component_distribution.log_prob(selected_action).sum(
                    dim=-1, keepdim=True
                )
                branch_entropies = component_distribution.entropy().sum(
                    dim=-1, keepdim=True
                )
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

    def distribution_diagnostics(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, float | None]:
        diagnostics: dict[str, float | None] = {}
        with torch.no_grad():
            if self.policy_architecture in {
                "simplex_branch_gaussian",
                "simplex_autoregressive_gaussian",
            }:
                log_stds = (
                    self.branch_log_stds
                    if self.policy_architecture == "simplex_branch_gaussian"
                    else self.autoregressive_branch_log_stds
                )
                for index, log_std in enumerate(log_stds):
                    number = index + 1
                    if not self.branch_train_mask[index]:
                        diagnostics[f"gaussian_policy_std_mean_branch_{number}"] = None
                        diagnostics[f"gaussian_policy_std_min_branch_{number}"] = None
                        diagnostics[f"gaussian_policy_std_max_branch_{number}"] = None
                        continue
                    std = torch.exp(torch.clamp(log_std, min=self.min_log_std))
                    diagnostics[f"gaussian_policy_std_mean_branch_{number}"] = float(
                        std.mean().item()
                    )
                    diagnostics[f"gaussian_policy_std_min_branch_{number}"] = float(
                        std.min().item()
                    )
                    diagnostics[f"gaussian_policy_std_max_branch_{number}"] = float(
                        std.max().item()
                    )
                return diagnostics

            if self.policy_architecture != "simplex_autoregressive_dirichlet":
                return diagnostics
            features = self.backbone(obs)
            provided = self._split_action(action, self.branch_sizes)
            previous_weights: list[torch.Tensor] = []
            concentration_range = (
                self.dirichlet_max_concentration
                - self.dirichlet_min_concentration
            )
            lower_threshold = (
                self.dirichlet_min_concentration + 0.01 * concentration_range
            )
            upper_threshold = (
                self.dirichlet_max_concentration - 0.01 * concentration_range
            )
            for index, size in enumerate(self.branch_sizes):
                number = index + 1
                branch_input = (
                    features
                    if not previous_weights
                    else torch.cat([features, *previous_weights], dim=-1)
                )
                if not self.branch_train_mask[index] or size == 1:
                    for name in (
                        "dirichlet_alpha0_mean",
                        "dirichlet_alpha0_min",
                        "dirichlet_alpha0_max",
                        "dirichlet_alpha_component_mean",
                        "dirichlet_alpha_component_min",
                        "dirichlet_alpha_component_max",
                        "dirichlet_alpha_lower_near_bound_rate",
                        "dirichlet_alpha_upper_near_bound_rate",
                    ):
                        diagnostics[f"{name}_branch_{number}"] = None
                    previous_weights.append(provided[index])
                    continue
                raw = self.autoregressive_dirichlet_heads[index](branch_input)
                concentration = self.dirichlet_min_concentration + (
                    self.dirichlet_max_concentration
                    - self.dirichlet_min_concentration
                ) * torch.sigmoid(raw)
                alpha0 = concentration.sum(dim=-1)
                diagnostics[f"dirichlet_alpha0_mean_branch_{number}"] = float(
                    alpha0.mean().item()
                )
                diagnostics[f"dirichlet_alpha0_min_branch_{number}"] = float(
                    alpha0.min().item()
                )
                diagnostics[f"dirichlet_alpha0_max_branch_{number}"] = float(
                    alpha0.max().item()
                )
                diagnostics[
                    f"dirichlet_alpha_component_mean_branch_{number}"
                ] = float(concentration.mean().item())
                diagnostics[
                    f"dirichlet_alpha_component_min_branch_{number}"
                ] = float(concentration.min().item())
                diagnostics[
                    f"dirichlet_alpha_component_max_branch_{number}"
                ] = float(concentration.max().item())
                diagnostics[
                    f"dirichlet_alpha_lower_near_bound_rate_branch_{number}"
                ] = float((concentration <= lower_threshold).float().mean().item())
                diagnostics[
                    f"dirichlet_alpha_upper_near_bound_rate_branch_{number}"
                ] = float((concentration >= upper_threshold).float().mean().item())
                previous_weights.append(provided[index])
        return diagnostics

    def branch_actor_parameter_groups(self) -> list[list[nn.Parameter]]:
        """Return actor-head parameters by CAOSD branch for diagnostics."""
        if not self.branch_sizes:
            return []
        if self.policy_architecture == "simplex_branch_gaussian":
            return [
                list(head.parameters()) + [log_std]
                for head, log_std in zip(
                    self.branch_mean_heads, self.branch_log_stds, strict=True
                )
            ]
        if self.policy_architecture == "simplex_autoregressive_gaussian":
            return [
                list(head.parameters()) + [log_std]
                for head, log_std in zip(
                    self.autoregressive_branch_mean_heads,
                    self.autoregressive_branch_log_stds,
                    strict=True,
                )
            ]
        if self.policy_architecture == "simplex_autoregressive_dirichlet":
            return [list(head.parameters()) for head in self.autoregressive_dirichlet_heads]
        return [[] for _ in self.branch_sizes]

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
