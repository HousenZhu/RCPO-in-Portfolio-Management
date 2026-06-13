from __future__ import annotations

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
) -> dict[str, float]:
    standalone_credit = model.branch_credit_mode == "standalone"
    advantages = _normalize_advantages(selected_advantages)
    normalized_branch_advantages = (
        _normalize_branch_advantages(branch_selected_advantages)
        if standalone_credit and branch_selected_advantages is not None
        else None
    )
    batch_size = batch.observations.shape[0]
    policy_losses: list[float] = []
    reward_value_losses: list[float] = []
    cost_value_losses: list[float] = []
    entropy_terms: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    stopped_by_target_kl = False
    optimizer_steps_completed = 0
    trigger_minibatch_kl: float | None = None
    branch_kl_terms: list[list[float]] = [list() for _ in model.branch_sizes]
    branch_entropy_terms: list[list[float]] = [list() for _ in model.branch_sizes]

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
                new_log_probs = output.log_prob
                entropy = output.entropy
                reward_values = output.reward_value
                cost_values = output.cost_value
            with profile_section(profiler, "loss_compute"):
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - optimization.clip_epsilon,
                    1.0 + optimization.clip_epsilon,
                )
                with torch.no_grad():
                    approx_kl = torch.mean(old_log_probs - new_log_probs)
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > optimization.clip_epsilon).float()
                    )
                if standalone_credit:
                    if minibatch_branch_advantages is None:
                        raise ValueError("Standalone credit requires branch advantages.")
                    branch_ratios = torch.exp(output.branch_log_probs - old_branch_log_probs)
                    clipped_branch_ratios = torch.clamp(
                        branch_ratios,
                        1.0 - optimization.clip_epsilon,
                        1.0 + optimization.clip_epsilon,
                    )
                    policy_loss = standalone_branch_policy_loss(
                        branch_ratios,
                        clipped_branch_ratios,
                        minibatch_branch_advantages,
                        branch_z_values,
                    )
                    reward_value_loss = torch.mean(
                        torch.square(branch_reward_returns - output.branch_reward_values)
                    )
                    cost_value_loss = torch.mean(
                        torch.square(branch_cost_returns - output.branch_cost_values)
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
                    reward_value_loss = torch.mean(torch.square(reward_returns - reward_values))
                    cost_value_loss = torch.mean(torch.square(cost_returns - cost_values))
                    entropy_bonus = torch.mean(entropy)

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
            if output.branch_log_probs.shape[1] == len(branch_kl_terms):
                branch_kls = torch.mean(
                    old_branch_log_probs - output.branch_log_probs,
                    dim=0,
                )
                branch_entropies = torch.mean(output.branch_entropies, dim=0)
                for branch_index in range(len(branch_kl_terms)):
                    branch_kl_terms[branch_index].append(float(branch_kls[branch_index].item()))
                    branch_entropy_terms[branch_index].append(
                        float(branch_entropies[branch_index].item())
                    )

            if (
                use_target_kl
                and optimization.target_kl is not None
                and float(approx_kl.item()) > optimization.target_kl
            ):
                stopped_by_target_kl = True
                trigger_minibatch_kl = float(approx_kl.item())
                break

        if stopped_by_target_kl:
            break

    metrics = {
        "policy_loss": float(np.mean(policy_losses)),
        "reward_value_loss": float(np.mean(reward_value_losses)),
        "cost_value_loss": float(np.mean(cost_value_losses)),
        "entropy": float(np.mean(entropy_terms)),
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "ppo_kl_early_stop": float(stopped_by_target_kl),
        "selected_advantage_mean": float(advantages.mean().item()),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "trigger_minibatch_kl": trigger_minibatch_kl,
    }
    for branch_index in range(4):
        number = branch_index + 1
        metrics[f"approx_kl_branch_{number}"] = (
            float(np.mean(branch_kl_terms[branch_index]))
            if branch_index < len(branch_kl_terms) and branch_kl_terms[branch_index]
            else 0.0
        )
        metrics[f"entropy_branch_{number}"] = (
            float(np.mean(branch_entropy_terms[branch_index]))
            if branch_index < len(branch_entropy_terms) and branch_entropy_terms[branch_index]
            else 0.0
        )
    return metrics


def update_ppo_actor_critic(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    optimization,
    profiler: TrainingProfiler | None = None,
) -> dict[str, float]:
    return _update_actor_critic_with_advantages(
        model=model,
        optimizer=optimizer,
        batch=batch,
        optimization=optimization,
        selected_advantages=batch.reward_advantages,
        branch_selected_advantages=(
            batch.branch_reward_advantages
            if model.branch_credit_mode == "standalone"
            else None
        ),
        train_cost_value=False,
        use_target_kl=True,
        profiler=profiler,
    )
