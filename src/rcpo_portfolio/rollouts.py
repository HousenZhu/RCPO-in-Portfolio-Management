from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import RewardNoiseConfig
from .env import PortfolioEnv
from .models import ActorCritic
from .profiling import TrainingProfiler, profile_section
from .reward_correction import RewardCorrector


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    log_probs: torch.Tensor
    true_rewards: torch.Tensor
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
    alpha_budget_ratio: float | None = None,
    drawdown_cost_scale: float | None = None,
    reward_noise_config: RewardNoiseConfig | None = None,
    reward_noise_rng: np.random.Generator | None = None,
    profiler: TrainingProfiler | None = None,
) -> RolloutBatch:
    device = device or next(model.parameters()).device
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    log_probs: list[float] = []
    true_rewards: list[float] = []
    costs: list[float] = []
    dones: list[float] = []
    reward_values: list[float] = []
    cost_values: list[float] = []
    episode_metrics: list[dict[str, float]] = []
    current_drawdowns: list[float] = []
    max_drawdowns: list[float] = []
    benchmark_current_drawdowns: list[float] = []
    benchmark_max_drawdowns: list[float] = []
    drawdown_benchmark_mode = ""
    effective_drawdown_budgets: list[float] = []
    alpha_targets: list[float] = []
    drawdown_gaps: list[float] = []
    drawdown_violations: list[float] = []
    drawdown_constraint_costs: list[float] = []
    allocation_constraint_1_violation_costs: list[float] = []
    allocation_constraint_2_violation_costs: list[float] = []
    allocation_constraint_1_weights: list[float] = []
    allocation_constraint_2_weights: list[float] = []
    simplex_z1_values: list[float] = []
    simplex_z2_values: list[float] = []
    simplex_z3_values: list[float] = []
    simplex_z4_values: list[float] = []
    concentrations: list[float] = []
    excess_concentration_costs: list[float] = []
    diversification_costs: list[float] = []

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_cost = 0.0
    episode_turnover = 0.0
    episode_steps = 0
    for _ in range(optimization.rollout_steps):
        with profile_section(profiler, "policy_action_forward"):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob_tensor, _, reward_value_tensor, cost_value_tensor = (
                    model.get_action_and_value(obs_tensor)
                )
        action = action_tensor.squeeze(0).cpu().numpy()
        with profile_section(profiler, "env_step"):
            next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        with profile_section(profiler, "rollout_storage_append"):
            observations.append(obs.astype(np.float32))
            actions.append(action.astype(np.float32))
            next_observations.append(next_obs.astype(np.float32))
            log_probs.append(float(log_prob_tensor.item()))
            true_rewards.append(float(reward))
            costs.append(float(info["constraint_cost"]))
            dones.append(float(done))
            reward_values.append(float(reward_value_tensor.item()))
            cost_values.append(float(cost_value_tensor.item()))
            current_drawdowns.append(float(info["current_drawdown"]))
            max_drawdowns.append(float(info["max_drawdown"]))
            benchmark_current_drawdowns.append(float(info["benchmark_current_drawdown"]))
            benchmark_max_drawdowns.append(float(info["benchmark_max_drawdown"]))
            drawdown_benchmark_mode = str(info["drawdown_benchmark_mode"])
            effective_drawdown_budgets.append(float(info["effective_drawdown_budget"]))
            if alpha_budget_ratio is not None:
                if drawdown_cost_scale is None:
                    raise ValueError("drawdown_cost_scale is required with alpha_budget_ratio.")
                alpha_targets.append(
                    float(
                        (alpha_budget_ratio * float(info["effective_drawdown_budget"])) ** 2
                        / max(drawdown_cost_scale, 1e-12)
                    )
                )
            drawdown_gaps.append(float(info["drawdown_gap"]))
            drawdown_violations.append(float(info["drawdown_violation"]))
            drawdown_constraint_costs.append(float(info["drawdown_constraint_cost"]))
            allocation_constraint_1_violation_costs.append(
                float(info["allocation_constraint_1_violation_cost"])
            )
            allocation_constraint_2_violation_costs.append(
                float(info["allocation_constraint_2_violation_cost"])
            )
            allocation_constraint_1_weights.append(
                float(info["allocation_constraint_1_weight"])
            )
            allocation_constraint_2_weights.append(
                float(info["allocation_constraint_2_weight"])
            )
            simplex_z1_values.append(float(info["simplex_z1"]))
            simplex_z2_values.append(float(info["simplex_z2"]))
            simplex_z3_values.append(float(info["simplex_z3"]))
            simplex_z4_values.append(float(info["simplex_z4"]))
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

    with profile_section(profiler, "policy_action_forward"):
        next_value_r, next_value_c = model.value(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        )
    with profile_section(profiler, "tensor_conversion"):
        observations_tensor = torch.as_tensor(
            np.asarray(observations), dtype=torch.float32, device=device
        )
        actions_tensor = torch.as_tensor(
            np.asarray(actions), dtype=torch.float32, device=device
        )
        next_observations_tensor = torch.as_tensor(
            np.asarray(next_observations), dtype=torch.float32, device=device
        )
        true_rewards_tensor = torch.as_tensor(true_rewards, dtype=torch.float32, device=device)
    reward_noise_enabled = bool(
        reward_noise_config is not None and reward_noise_config.enabled
    )
    reward_noise_std = (
        float(reward_noise_config.std)
        if reward_noise_config is not None
        else 0.0
    )
    with profile_section(profiler, "reward_noise"):
        if reward_noise_enabled and reward_noise_std > 0.0:
            if reward_noise_rng is None:
                raise ValueError("reward_noise_rng is required when reward noise is enabled.")
            reward_noise = reward_noise_rng.normal(
                loc=0.0,
                scale=reward_noise_std,
                size=len(true_rewards),
            ).astype(np.float32)
        else:
            reward_noise = np.zeros(len(true_rewards), dtype=np.float32)
        reward_noise_tensor = torch.as_tensor(reward_noise, dtype=torch.float32, device=device)
        observed_rewards_tensor = true_rewards_tensor + reward_noise_tensor
    with profile_section(profiler, "reward_correction"):
        correction = reward_corrector.update_and_correct(
            observations_tensor,
            actions_tensor,
            next_observations_tensor,
            observed_rewards_tensor,
        )
        rewards_tensor = correction.corrected_rewards.to(dtype=torch.float32)
    with profile_section(profiler, "tensor_conversion"):
        costs_tensor = torch.as_tensor(costs, dtype=torch.float32, device=device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
        reward_values_tensor = torch.as_tensor(
            reward_values, dtype=torch.float32, device=device
        )
        cost_values_tensor = torch.as_tensor(
            cost_values, dtype=torch.float32, device=device
        )
    with profile_section(profiler, "reward_gae"):
        reward_advantages, reward_returns = compute_gae(
            rewards_tensor,
            reward_values_tensor,
            dones_tensor,
            next_value_r.squeeze(0).detach(),
            optimization.gamma,
            optimization.gae_lambda,
        )
    with profile_section(profiler, "cost_gae"):
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
        "batch_true_reward_mean": float(true_rewards_tensor.mean().item()),
        "batch_observed_reward_mean": float(observed_rewards_tensor.mean().item()),
        "batch_reward_noise_mean": float(reward_noise_tensor.mean().item()),
        "batch_reward_noise_std": float(reward_noise_tensor.std(unbiased=False).item()),
        "reward_noise_enabled": int(reward_noise_enabled),
        "reward_noise_std": reward_noise_std,
        "batch_constraint_cost_mean": float(np.mean(costs)),
        "batch_current_drawdown_mean": float(np.mean(current_drawdowns)),
        "batch_max_drawdown_mean": float(np.mean(max_drawdowns)),
        "batch_benchmark_current_drawdown_mean": float(
            np.mean(benchmark_current_drawdowns)
        ),
        "batch_benchmark_max_drawdown_mean": float(np.mean(benchmark_max_drawdowns)),
        "drawdown_benchmark_mode": drawdown_benchmark_mode,
        "batch_effective_drawdown_budget_mean": float(
            np.mean(effective_drawdown_budgets)
        ),
        "batch_alpha_target_mean": float(np.mean(alpha_targets)) if alpha_targets else 0.0,
        "batch_drawdown_gap_mean": float(np.mean(drawdown_gaps)),
        "batch_drawdown_violation_mean": float(np.mean(drawdown_violations)),
        "batch_drawdown_constraint_cost_mean": float(np.mean(drawdown_constraint_costs)),
        "batch_allocation_constraint_1_violation_cost_mean": float(
            np.mean(allocation_constraint_1_violation_costs)
        ),
        "batch_allocation_constraint_2_violation_cost_mean": float(
            np.mean(allocation_constraint_2_violation_costs)
        ),
        "batch_allocation_constraint_1_weight_mean": float(
            np.mean(allocation_constraint_1_weights)
        ),
        "batch_allocation_constraint_2_weight_mean": float(
            np.mean(allocation_constraint_2_weights)
        ),
        "batch_simplex_z1_mean": float(np.mean(simplex_z1_values)),
        "batch_simplex_z2_mean": float(np.mean(simplex_z2_values)),
        "batch_simplex_z3_mean": float(np.mean(simplex_z3_values)),
        "batch_simplex_z4_mean": float(np.mean(simplex_z4_values)),
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
        true_rewards=true_rewards_tensor,
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
