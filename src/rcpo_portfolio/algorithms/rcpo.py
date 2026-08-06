from __future__ import annotations

import numpy as np
import torch

from ..models import ActorCritic
from ..profiling import TrainingProfiler, profile_section
from ..rollouts import RolloutBatch
from .ppo import _update_actor_critic_with_advantages


def combine_advantages(
    reward_advantages: torch.Tensor,
    cost_advantages: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    return reward_advantages - float(lambda_value) * cost_advantages


def update_lagrange_multiplier(
    lambda_value: float,
    observed_cost: float,
    alpha: float,
    learning_rate: float | None = None,
    learning_rate_up: float | None = None,
    learning_rate_down: float | None = None,
) -> float:
    gap = float(observed_cost - alpha)
    if learning_rate_up is None or learning_rate_down is None:
        if learning_rate is None:
            raise ValueError("A lambda learning rate is required.")
        learning_rate_up = learning_rate
        learning_rate_down = learning_rate
    selected_learning_rate = learning_rate_up if gap > 0.0 else learning_rate_down
    return max(0.0, float(lambda_value + float(selected_learning_rate) * gap))


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(values))).item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(
        torch.sum(left_centered**2) * torch.sum(right_centered**2)
    )
    if float(denominator.item()) <= 1e-12:
        return None
    return float(torch.sum(left_centered * right_centered).item() / denominator.item())


def update_rcpo_actor_critic(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    optimization,
    lambda_value: float,
    alpha: float | None,
    lambda_lr: float,
    lambda_lr_up: float | None = None,
    lambda_lr_down: float | None = None,
    profiler: TrainingProfiler | None = None,
) -> tuple[dict[str, float | int | None], float, list[float]]:
    lambda_before = float(lambda_value)
    combined_advantages = combine_advantages(
        batch.reward_advantages,
        batch.cost_advantages,
        lambda_before,
    )
    if model.branch_credit_mode == "standalone":
        branch_cost_advantages = batch.branch_cost_advantages
    elif model.branch_credit_mode == "standalone_reward_global_cost":
        branch_cost_advantages = batch.cost_advantages.unsqueeze(-1).expand_as(
            batch.branch_reward_advantages
        )
    else:
        branch_cost_advantages = None
    branch_combined_advantages = (
        combine_advantages(
            batch.branch_reward_advantages,
            branch_cost_advantages,
            lambda_before,
        )
        if branch_cost_advantages is not None
        else None
    )
    metrics = _update_actor_critic_with_advantages(
        model=model,
        optimizer=optimizer,
        batch=batch,
        optimization=optimization,
        selected_advantages=combined_advantages,
        branch_selected_advantages=branch_combined_advantages,
        train_cost_value=True,
        use_target_kl=True,
        profiler=profiler,
    )
    metrics["combined_advantage_mean"] = float(combined_advantages.mean().item())

    if branch_combined_advantages is not None and branch_cost_advantages is not None:
        for branch_index in range(branch_combined_advantages.shape[1]):
            number = branch_index + 1
            reward_advantage = batch.branch_reward_advantages[:, branch_index]
            cost_advantage = branch_cost_advantages[:, branch_index]
            metrics[f"branch_lambda_cost_adv_ratio_{number}"] = (
                abs(lambda_before) * _rms(cost_advantage)
                / max(_rms(reward_advantage), 1e-12)
            )
            metrics[f"branch_combined_advantage_std_{number}"] = float(
                branch_combined_advantages[:, branch_index]
                .std(unbiased=False)
                .item()
            )
            metrics[f"branch_reward_cost_adv_correlation_{number}"] = _correlation(
                reward_advantage,
                cost_advantage,
            )

    lambda_updates: list[float] = []
    if alpha is not None:
        with profile_section(profiler, "lambda_update"):
            lambda_value = update_lagrange_multiplier(
                lambda_before,
                float(batch.info_summary["batch_constraint_cost_mean"]),
                alpha,
                learning_rate=lambda_lr,
                learning_rate_up=lambda_lr_up,
                learning_rate_down=lambda_lr_down,
            )
            lambda_updates.append(float(lambda_value))
    metrics["lambda_before"] = lambda_before
    metrics["lambda_after"] = float(lambda_value)
    metrics["lambda_delta"] = float(lambda_value - lambda_before)
    metrics["lambda_update_count"] = len(lambda_updates)
    return metrics, float(lambda_value), lambda_updates
