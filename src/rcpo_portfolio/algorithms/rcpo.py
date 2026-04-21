from __future__ import annotations

import torch

from ..models import ActorCritic
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
    learning_rate: float,
) -> float:
    return max(0.0, float(lambda_value + learning_rate * (observed_cost - alpha)))


def update_rcpo_actor_critic(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    optimization,
    lambda_value: float,
    alpha: float | None,
    lambda_lr: float,
) -> tuple[dict[str, float], float, list[float]]:
    combined_advantages = combine_advantages(
        batch.reward_advantages,
        batch.cost_advantages,
        lambda_value,
    )
    metrics = _update_actor_critic_with_advantages(
        model=model,
        optimizer=optimizer,
        batch=batch,
        optimization=optimization,
        selected_advantages=combined_advantages,
        train_cost_value=True,
        use_target_kl=True,
    )
    metrics["combined_advantage_mean"] = float(combined_advantages.mean().item())

    lambda_updates: list[float] = []
    if alpha is not None:
        for _ in range(optimization.epochs):
            lambda_value = update_lagrange_multiplier(
                lambda_value,
                float(batch.info_summary["batch_constraint_cost_mean"]),
                alpha,
                lambda_lr,
            )
            lambda_updates.append(float(lambda_value))
    return metrics, float(lambda_value), lambda_updates
