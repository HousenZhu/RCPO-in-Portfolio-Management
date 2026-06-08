from __future__ import annotations

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
        profiler=profiler,
    )
    metrics["combined_advantage_mean"] = float(combined_advantages.mean().item())

    lambda_updates: list[float] = []
    if alpha is not None:
        with profile_section(profiler, "lambda_update"):
            for _ in range(optimization.epochs):
                lambda_value = update_lagrange_multiplier(
                    lambda_value,
                    float(batch.info_summary["batch_constraint_cost_mean"]),
                    alpha,
                    learning_rate=lambda_lr,
                    learning_rate_up=lambda_lr_up,
                    learning_rate_down=lambda_lr_down,
                )
                lambda_updates.append(float(lambda_value))
    return metrics, float(lambda_value), lambda_updates
