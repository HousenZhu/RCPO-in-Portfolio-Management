from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..models import ActorCritic
from ..profiling import TrainingProfiler, profile_section
from ..rollouts import RolloutBatch


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def _normalize_branch_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean(dim=0, keepdim=True)) / (
        advantages.std(dim=0, unbiased=False, keepdim=True) + 1e-8
    )


def _explained_variance(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    epsilon: float = 1e-12,
) -> tuple[float | None, float]:
    target_variance = float(torch.var(targets, unbiased=False).item())
    if target_variance <= epsilon:
        return None, target_variance
    residual_variance = float(torch.var(targets - predictions, unbiased=False).item())
    return float(1.0 - residual_variance / target_variance), target_variance


def _gradient_norm(parameters) -> float | None:
    squared_norm = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        squared_norm += float(torch.sum(parameter.grad.detach() ** 2).item())
    return math.sqrt(squared_norm) if found else None


def standalone_branch_policy_loss(
    branch_ratios: torch.Tensor,
    clipped_branch_ratios: torch.Tensor,
    branch_advantages: torch.Tensor,
    branch_z_values: torch.Tensor,
) -> torch.Tensor:
    """Return the CAOSD-mass-weighted PPO loss for branch-local advantages."""
    branch_surrogate = torch.min(
        branch_ratios * branch_advantages,
        clipped_branch_ratios * branch_advantages,
    )
    return -torch.mean(torch.sum(branch_z_values * branch_surrogate, dim=-1))


def _update_actor_critic_with_advantages(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    optimization,
    selected_advantages: torch.Tensor,
    branch_selected_advantages: torch.Tensor | None,
    train_cost_value: bool,
    use_target_kl: bool,
    profiler: TrainingProfiler | None = None,
) -> dict[str, float | int | None]:
    standalone_credit = model.branch_credit_mode != "global"
    advantages = _normalize_advantages(selected_advantages)
    normalized_branch_advantages = (
        _normalize_branch_advantages(branch_selected_advantages)
        if standalone_credit and branch_selected_advantages is not None
        else None
    )
    reward_critic_ev, reward_target_variance = _explained_variance(
        batch.reward_returns,
        batch.reward_values,
    )
    cost_critic_ev, cost_target_variance = (
        _explained_variance(batch.cost_returns, batch.cost_values)
        if train_cost_value
        else (None, float(torch.var(batch.cost_returns, unbiased=False).item()))
    )
    branch_reward_evs: list[float | None] = []
    branch_reward_target_variances: list[float | None] = []
    for branch_index in range(len(model.branch_sizes)):
        if not standalone_credit or not model.branch_train_mask[branch_index]:
            branch_reward_evs.append(None)
            branch_reward_target_variances.append(None)
            continue
        ev, variance = _explained_variance(
            batch.branch_reward_returns[:, branch_index],
            batch.branch_reward_values[:, branch_index],
        )
        branch_reward_evs.append(ev)
        branch_reward_target_variances.append(variance)

    batch_size = batch.observations.shape[0]
    policy_losses: list[float] = []
    reward_value_losses: list[float] = []
    cost_value_losses: list[float] = []
    entropy_terms: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    stopped_by_target_kl = False
    optimizer_steps_attempted = 0
    optimizer_steps_completed = 0
    rejected_minibatch_kl: float | None = None
    branch_kl_terms: list[list[float]] = [list() for _ in model.branch_sizes]
    branch_entropy_terms: list[list[float]] = [list() for _ in model.branch_sizes]
    branch_clip_terms: list[list[float]] = [list() for _ in model.branch_sizes]
    actor_gradient_norms: list[float] = []
    backbone_gradient_norms: list[float] = []
    branch_gradient_norms: list[list[float]] = [list() for _ in model.branch_sizes]

    for _ in range(optimization.epochs):
        with profile_section(profiler, "minibatch_indexing"):
            permutation = torch.randperm(batch_size, device=batch.observations.device)
        for start in range(0, batch_size, optimization.minibatch_size):
            with profile_section(profiler, "minibatch_indexing"):
                batch_indices = permutation[start : start + optimization.minibatch_size]
                observations = batch.observations[batch_indices]
                actions = batch.actions[batch_indices]
                old_log_probs = batch.log_probs[batch_indices]
                reward_returns = batch.reward_returns[batch_indices]
                cost_returns = batch.cost_returns[batch_indices]
                minibatch_advantages = advantages[batch_indices]
                minibatch_branch_advantages = (
                    normalized_branch_advantages[batch_indices]
                    if normalized_branch_advantages is not None
                    else None
                )
                old_branch_log_probs = batch.branch_log_probs[batch_indices]
                branch_z_values = batch.branch_z_values[batch_indices]
                branch_reward_returns = batch.branch_reward_returns[batch_indices]
                branch_cost_returns = batch.branch_cost_returns[batch_indices]

            with profile_section(profiler, "model_recompute_forward"):
                output = model.get_policy_output(observations, action=actions)
            with profile_section(profiler, "loss_compute"):
                log_ratio = torch.clamp(output.log_prob - old_log_probs, -20.0, 20.0)
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - optimization.clip_epsilon,
                    1.0 + optimization.clip_epsilon,
                )
                # The standalone PPO surrogate needs differentiable branch ratios.
                # Only the KL and clipping diagnostics are detached.
                branch_log_ratio = torch.clamp(
                    output.branch_log_probs - old_branch_log_probs,
                    -20.0,
                    20.0,
                )
                branch_ratios = torch.exp(branch_log_ratio)
                clipped_branch_ratios = torch.clamp(
                    branch_ratios,
                    1.0 - optimization.clip_epsilon,
                    1.0 + optimization.clip_epsilon,
                )
                with torch.no_grad():
                    approx_kl = torch.mean(ratio - 1.0 - log_ratio)
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > optimization.clip_epsilon).float()
                    )
                    branch_kls = torch.mean(
                        branch_ratios.detach() - 1.0 - branch_log_ratio.detach(),
                        dim=0,
                    )
                    branch_clip_fractions = torch.mean(
                        (
                            torch.abs(branch_ratios.detach() - 1.0)
                            > optimization.clip_epsilon
                        ).float(),
                        dim=0,
                    )
                optimizer_steps_attempted += 1
                if (
                    use_target_kl
                    and optimization.target_kl is not None
                    and float(approx_kl.item()) > optimization.target_kl
                ):
                    stopped_by_target_kl = True
                    rejected_minibatch_kl = float(approx_kl.item())
                    break

                if standalone_credit:
                    if minibatch_branch_advantages is None:
                        raise ValueError("Standalone credit requires branch advantages.")
                    policy_loss = standalone_branch_policy_loss(
                        branch_ratios,
                        clipped_branch_ratios,
                        minibatch_branch_advantages,
                        branch_z_values,
                    )
                    reward_value_loss = torch.mean(
                        torch.square(branch_reward_returns - output.branch_reward_values)
                    )
                    if model.branch_credit_mode == "standalone":
                        cost_value_loss = torch.mean(
                            torch.square(branch_cost_returns - output.branch_cost_values)
                        )
                    else:
                        cost_value_loss = torch.mean(
                            torch.square(cost_returns - output.cost_value)
                        )
                    entropy_bonus = torch.mean(
                        torch.sum(branch_z_values * output.branch_entropies, dim=-1)
                    )
                else:
                    policy_loss = -torch.mean(
                        torch.min(
                            ratio * minibatch_advantages,
                            clipped_ratio * minibatch_advantages,
                        )
                    )
                    reward_value_loss = torch.mean(
                        torch.square(reward_returns - output.reward_value)
                    )
                    cost_value_loss = torch.mean(
                        torch.square(cost_returns - output.cost_value)
                    )
                    entropy_bonus = torch.mean(output.entropy)

                total_loss = (
                    policy_loss
                    + optimization.reward_value_coef * reward_value_loss
                    + (optimization.cost_value_coef if train_cost_value else 0.0)
                    * cost_value_loss
                    - optimization.entropy_coef * entropy_bonus
                )

            with profile_section(profiler, "backward"):
                optimizer.zero_grad()
                total_loss.backward()
                actor_gradient_norm = _gradient_norm(
                    parameter
                    for name, parameter in model.named_parameters()
                    if "value" not in name
                )
                backbone_gradient_norm = _gradient_norm(model.backbone.parameters())
                if actor_gradient_norm is not None:
                    actor_gradient_norms.append(actor_gradient_norm)
                if backbone_gradient_norm is not None:
                    backbone_gradient_norms.append(backbone_gradient_norm)
                head_groups = model.branch_actor_parameter_groups()
                for branch_index, parameters in enumerate(head_groups):
                    norm = _gradient_norm(parameters)
                    if norm is not None:
                        branch_gradient_norms[branch_index].append(norm)
                nn.utils.clip_grad_norm_(model.parameters(), optimization.max_grad_norm)
            with profile_section(profiler, "optimizer_step"):
                optimizer.step()
                optimizer_steps_completed += 1

            policy_losses.append(float(policy_loss.item()))
            reward_value_losses.append(float(reward_value_loss.item()))
            cost_value_losses.append(float(cost_value_loss.item()))
            entropy_terms.append(float(entropy_bonus.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))
            branch_entropies = torch.mean(output.branch_entropies, dim=0)
            for branch_index in range(len(branch_kl_terms)):
                if not model.branch_train_mask[branch_index]:
                    continue
                branch_kl_terms[branch_index].append(float(branch_kls[branch_index].item()))
                branch_entropy_terms[branch_index].append(
                    float(branch_entropies[branch_index].item())
                )
                branch_clip_terms[branch_index].append(
                    float(branch_clip_fractions[branch_index].item())
                )

        if stopped_by_target_kl:
            break

    metrics: dict[str, float | int | None] = {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "reward_value_loss": float(np.mean(reward_value_losses)) if reward_value_losses else 0.0,
        "cost_value_loss": float(np.mean(cost_value_losses)) if cost_value_losses else 0.0,
        "entropy": float(np.mean(entropy_terms)) if entropy_terms else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "ppo_kl_early_stop": int(stopped_by_target_kl),
        "selected_advantage_mean": float(advantages.mean().item()),
        "optimizer_steps_attempted": int(optimizer_steps_attempted),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "rejected_minibatch_kl": rejected_minibatch_kl,
        "trigger_minibatch_kl": rejected_minibatch_kl,
        "actor_gradient_norm": (
            float(np.mean(actor_gradient_norms)) if actor_gradient_norms else None
        ),
        "backbone_gradient_norm": (
            float(np.mean(backbone_gradient_norms)) if backbone_gradient_norms else None
        ),
        "reward_critic_ev": reward_critic_ev,
        "reward_critic_target_variance": reward_target_variance,
        "cost_critic_ev": cost_critic_ev,
        "cost_critic_target_variance": cost_target_variance,
    }
    for branch_index in range(4):
        number = branch_index + 1
        active = (
            branch_index < len(model.branch_train_mask)
            and model.branch_train_mask[branch_index]
        )
        metrics[f"approx_kl_branch_{number}"] = (
            float(np.mean(branch_kl_terms[branch_index]))
            if active and branch_kl_terms[branch_index]
            else None
        )
        metrics[f"entropy_branch_{number}"] = (
            float(np.mean(branch_entropy_terms[branch_index]))
            if active and branch_entropy_terms[branch_index]
            else None
        )
        metrics[f"clip_fraction_branch_{number}"] = (
            float(np.mean(branch_clip_terms[branch_index]))
            if active and branch_clip_terms[branch_index]
            else None
        )
        metrics[f"actor_gradient_norm_branch_{number}"] = (
            float(np.mean(branch_gradient_norms[branch_index]))
            if active and branch_gradient_norms[branch_index]
            else None
        )
        metrics[f"branch_reward_critic_ev_{number}"] = (
            branch_reward_evs[branch_index]
            if branch_index < len(branch_reward_evs)
            else None
        )
        metrics[f"branch_reward_critic_target_variance_{number}"] = (
            branch_reward_target_variances[branch_index]
            if branch_index < len(branch_reward_target_variances)
            else None
        )
        metrics[f"branch_cost_critic_ev_{number}"] = None
    return metrics


def update_ppo_actor_critic(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    optimization,
    profiler: TrainingProfiler | None = None,
) -> dict[str, float | int | None]:
    return _update_actor_critic_with_advantages(
        model=model,
        optimizer=optimizer,
        batch=batch,
        optimization=optimization,
        selected_advantages=batch.reward_advantages,
        branch_selected_advantages=(
            batch.branch_reward_advantages
            if model.branch_credit_mode != "global"
            else None
        ),
        train_cost_value=False,
        use_target_kl=True,
        profiler=profiler,
    )
