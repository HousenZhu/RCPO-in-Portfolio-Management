from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env import PortfolioEnv
from .models import ActorCritic
from .reward_correction import RewardCorrector


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    log_probs: torch.Tensor
    observed_rewards: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor
    reward_returns: torch.Tensor
    cost_returns: torch.Tensor
    reward_advantages: torch.Tensor
    cost_advantages: torch.Tensor
    info_summary: dict[str, float | int | str]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[index]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[index]
            next_values = values[index + 1]
        delta = rewards[index] + gamma * next_values * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return advantages, returns


def _flatten_metrics(metrics: list[dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    return float(np.mean([metric[key] for metric in metrics]))


def collect_rollout(
    env: PortfolioEnv,
    model: ActorCritic,
    optimization,
    reward_corrector: RewardCorrector,
    device: torch.device | None = None,
) -> RolloutBatch:
    device = device or next(model.parameters()).device
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    log_probs: list[float] = []
    observed_rewards: list[float] = []
    costs: list[float] = []
    dones: list[float] = []
    reward_values: list[float] = []
    cost_values: list[float] = []
    episode_metrics: list[dict[str, float]] = []
    downside_costs: list[float] = []
    normalized_downside_costs: list[float] = []
    sortino_violation_costs: list[float] = []
    sortino_ratios: list[float] = []
    group_a_min_violation_costs: list[float] = []
    group_b_max_violation_costs: list[float] = []
    group_a_weights: list[float] = []
    group_b_weights: list[float] = []
    concentrations: list[float] = []
    excess_concentration_costs: list[float] = []
    diversification_costs: list[float] = []

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_cost = 0.0
    episode_turnover = 0.0
    episode_steps = 0
    for _ in range(optimization.rollout_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_prob_tensor, _, reward_value_tensor, cost_value_tensor = (
                model.get_action_and_value(obs_tensor)
            )
        action = action_tensor.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        observations.append(obs.astype(np.float32))
        actions.append(action.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))
        log_probs.append(float(log_prob_tensor.item()))
        observed_rewards.append(float(reward))
        costs.append(float(info["constraint_cost"]))
        dones.append(float(done))
        reward_values.append(float(reward_value_tensor.item()))
        cost_values.append(float(cost_value_tensor.item()))
        downside_costs.append(float(info["downside_cost"]))
        normalized_downside_costs.append(float(info["normalized_downside_cost"]))
        sortino_violation_costs.append(float(info["sortino_violation_cost"]))
        sortino_ratios.append(float(info["sortino_ratio"]))
        group_a_min_violation_costs.append(float(info["group_a_min_violation_cost"]))
        group_b_max_violation_costs.append(float(info["group_b_max_violation_cost"]))
        group_a_weights.append(float(info["group_a_weight"]))
        group_b_weights.append(float(info["group_b_weight"]))
        concentrations.append(float(info["concentration"]))
        excess_concentration_costs.append(float(info["excess_concentration_cost"]))
        diversification_costs.append(float(info["diversification_cost"]))

        episode_reward += float(info["net_return"])
        episode_cost += float(info["constraint_cost"])
        episode_turnover += float(info["turnover"])
        episode_steps += 1

        obs = next_obs
        if done:
            episode_metrics.append(
                {
                    "episode_return": episode_reward,
                    "episode_cost": episode_cost / max(episode_steps, 1),
                    "episode_turnover": episode_turnover / max(episode_steps, 1),
                }
            )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_turnover = 0.0
            episode_steps = 0

    next_value_r, next_value_c = model.value(
        torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    )
    observations_tensor = torch.as_tensor(
        np.asarray(observations), dtype=torch.float32, device=device
    )
    actions_tensor = torch.as_tensor(
        np.asarray(actions), dtype=torch.float32, device=device
    )
    next_observations_tensor = torch.as_tensor(
        np.asarray(next_observations), dtype=torch.float32, device=device
    )
    observed_rewards_tensor = torch.as_tensor(
        observed_rewards, dtype=torch.float32, device=device
    )
    correction = reward_corrector.update_and_correct(
        observations_tensor,
        actions_tensor,
        next_observations_tensor,
        observed_rewards_tensor,
    )
    rewards_tensor = correction.corrected_rewards.to(dtype=torch.float32)
    costs_tensor = torch.as_tensor(costs, dtype=torch.float32, device=device)
    dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
    reward_values_tensor = torch.as_tensor(
        reward_values, dtype=torch.float32, device=device
    )
    cost_values_tensor = torch.as_tensor(
        cost_values, dtype=torch.float32, device=device
    )
    reward_advantages, reward_returns = compute_gae(
        rewards_tensor,
        reward_values_tensor,
        dones_tensor,
        next_value_r.squeeze(0).detach(),
        optimization.gamma,
        optimization.gae_lambda,
    )
    cost_advantages, cost_returns = compute_gae(
        costs_tensor,
        cost_values_tensor,
        dones_tensor,
        next_value_c.squeeze(0).detach(),
        optimization.gamma,
        optimization.gae_lambda,
    )

    info_summary: dict[str, float | int | str] = {
        "batch_reward_mean": float(rewards_tensor.mean().item()),
        "batch_observed_reward_mean": float(observed_rewards_tensor.mean().item()),
        "batch_constraint_cost_mean": float(np.mean(costs)),
        "batch_downside_cost_mean": float(np.mean(downside_costs)),
        "batch_normalized_downside_cost_mean": float(np.mean(normalized_downside_costs)),
        "batch_sortino_violation_cost_mean": float(np.mean(sortino_violation_costs)),
        "batch_sortino_ratio_mean": float(np.mean(sortino_ratios)),
        "batch_group_a_min_violation_cost_mean": float(np.mean(group_a_min_violation_costs)),
        "batch_group_b_max_violation_cost_mean": float(np.mean(group_b_max_violation_costs)),
        "batch_group_a_weight_mean": float(np.mean(group_a_weights)),
        "batch_group_b_weight_mean": float(np.mean(group_b_weights)),
        "batch_concentration_mean": float(np.mean(concentrations)),
        "batch_excess_concentration_cost_mean": float(np.mean(excess_concentration_costs)),
        "batch_diversification_cost_mean": float(np.mean(diversification_costs)),
        "batch_turnover_mean": _flatten_metrics(episode_metrics, "episode_turnover"),
        "episode_return_mean": _flatten_metrics(episode_metrics, "episode_return"),
        "episode_cost_mean": _flatten_metrics(episode_metrics, "episode_cost"),
        **correction.metrics,
    }
    return RolloutBatch(
        observations=observations_tensor,
        actions=actions_tensor,
        next_observations=next_observations_tensor,
        log_probs=torch.as_tensor(log_probs, dtype=torch.float32, device=device),
        observed_rewards=observed_rewards_tensor,
        rewards=rewards_tensor.detach(),
        costs=costs_tensor,
        dones=dones_tensor,
        reward_values=reward_values_tensor,
        cost_values=cost_values_tensor,
        reward_returns=reward_returns.detach(),
        cost_returns=cost_returns.detach(),
        reward_advantages=reward_advantages.detach(),
        cost_advantages=cost_advantages.detach(),
        info_summary=info_summary,
    )
