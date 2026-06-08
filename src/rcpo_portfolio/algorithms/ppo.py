from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..models import ActorCritic
from ..profiling import TrainingProfiler, profile_section
from ..rollouts import RolloutBatch


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def _update_actor_critic_with_advantages(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    optimization,
    selected_advantages: torch.Tensor,
    train_cost_value: bool,
    use_target_kl: bool,
    profiler: TrainingProfiler | None = None,
) -> dict[str, float]:
    advantages = _normalize_advantages(selected_advantages)
    batch_size = batch.observations.shape[0]
    policy_losses: list[float] = []
    reward_value_losses: list[float] = []
    cost_value_losses: list[float] = []
    entropy_terms: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    stopped_by_target_kl = False

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

            with profile_section(profiler, "model_recompute_forward"):
                _, new_log_probs, entropy, reward_values, cost_values = (
                    model.get_action_and_value(observations, action=actions)
                )
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

            policy_losses.append(float(policy_loss.item()))
            reward_value_losses.append(float(reward_value_loss.item()))
            cost_value_losses.append(float(cost_value_loss.item()))
            entropy_terms.append(float(entropy_bonus.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))

            if (
                use_target_kl
                and optimization.target_kl is not None
                and float(approx_kl.item()) > optimization.target_kl
            ):
                stopped_by_target_kl = True
                break

        if stopped_by_target_kl:
            break

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "reward_value_loss": float(np.mean(reward_value_losses)),
        "cost_value_loss": float(np.mean(cost_value_losses)),
        "entropy": float(np.mean(entropy_terms)),
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "ppo_kl_early_stop": float(stopped_by_target_kl),
        "selected_advantage_mean": float(advantages.mean().item()),
    }


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
        train_cost_value=False,
        use_target_kl=True,
        profiler=profiler,
    )
