from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .branch_credit import (
    uses_counterfactual_context,
    uses_counterfactual_cost,
    uses_counterfactual_reward,
)
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
    branch_log_probs: torch.Tensor
    branch_entropies: torch.Tensor
    branch_rewards: torch.Tensor
    branch_costs: torch.Tensor
    branch_z_values: torch.Tensor
    branch_reward_values: torch.Tensor
    branch_cost_values: torch.Tensor
    branch_reward_returns: torch.Tensor
    branch_cost_returns: torch.Tensor
    branch_reward_advantages: torch.Tensor
    branch_cost_advantages: torch.Tensor
    info_summary: dict[str, float | int | str]
    branch_critic_contexts: torch.Tensor | None = None
    episode_constraint_gaps: tuple[float, ...] = ()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
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
    fixed_alpha_target: float | None = None,
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
    branch_log_probs: list[np.ndarray] = []
    branch_entropies: list[np.ndarray] = []
    branch_rewards: list[np.ndarray] = []
    branch_costs: list[np.ndarray] = []
    branch_z_values: list[np.ndarray] = []
    branch_reward_values: list[np.ndarray] = []
    branch_cost_values: list[np.ndarray] = []
    branch_turnovers: list[np.ndarray] = []
    branch_transaction_costs: list[np.ndarray] = []
    branch_max_drawdowns: list[np.ndarray] = []
    branch_critic_contexts: list[np.ndarray] = []
    branch_actual_rewards: list[np.ndarray] = []
    branch_counterfactual_rewards: list[np.ndarray] = []
    branch_delta_rewards: list[np.ndarray] = []
    branch_actual_costs: list[np.ndarray] = []
    branch_counterfactual_costs: list[np.ndarray] = []
    branch_delta_costs: list[np.ndarray] = []
    counterfactual_weight_distances: list[np.ndarray] = []
    counterfactual_turnover_differences: list[np.ndarray] = []
    counterfactual_drawdown_differences: list[np.ndarray] = []
    counterfactual_zero_effects: list[np.ndarray] = []
    counterfactual_nonfinite_count = 0
    counterfactual_mapping_failure_count = 0
    episode_metrics: list[dict[str, float]] = []
    current_drawdowns: list[float] = []
    max_drawdowns: list[float] = []
    benchmark_current_drawdowns: list[float] = []
    benchmark_max_drawdowns: list[float] = []
    branch_train_mask: np.ndarray | None = None
    drawdown_benchmark_mode = ""
    effective_drawdown_budgets: list[float] = []
    alpha_targets: list[float] = []
    drawdown_gaps: list[float] = []
    drawdown_violations: list[float] = []
    drawdown_constraint_costs: list[float] = []
    allocation_constraint_1_violation_costs: list[float] = []
    allocation_constraint_2_violation_costs: list[float] = []
    allocation_constraint_raw_costs: list[float] = []
    allocation_constraint_costs: list[float] = []
    allocation_drawdown_constraint_costs: list[float] = []
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
    episode_alpha_target = 0.0
    episode_turnover = 0.0
    episode_steps = 0
    for _ in range(optimization.rollout_steps):
        with profile_section(profiler, "policy_action_forward"):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            critic_context = (
                env.counterfactual_critic_context()
                if uses_counterfactual_context(model.branch_credit_mode)
                else np.zeros((4, 0), dtype=np.float32)
            )
            critic_context_tensor = (
                torch.as_tensor(
                    critic_context, dtype=torch.float32, device=device
                ).unsqueeze(0)
                if uses_counterfactual_context(model.branch_credit_mode)
                else None
            )
            with torch.no_grad():
                policy_output = model.get_policy_output(
                    obs_tensor,
                    counterfactual_context=critic_context_tensor,
                )
                action_tensor = policy_output.action
                log_prob_tensor = policy_output.log_prob
                reward_value_tensor = policy_output.reward_value
                cost_value_tensor = policy_output.cost_value
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
            branch_log_probs.append(
                policy_output.branch_log_probs.squeeze(0).cpu().numpy().astype(np.float32)
            )
            branch_entropies.append(
                policy_output.branch_entropies.squeeze(0).cpu().numpy().astype(np.float32)
            )
            branch_reward_values.append(
                policy_output.branch_reward_values.squeeze(0).cpu().numpy().astype(np.float32)
            )
            branch_cost_values.append(
                policy_output.branch_cost_values.squeeze(0).cpu().numpy().astype(np.float32)
            )
            branch_critic_contexts.append(critic_context)
            actual_branch_rewards = np.asarray(info["branch_rewards"], dtype=np.float32)
            actual_branch_costs = np.asarray(info["branch_costs"], dtype=np.float32)
            counterfactual_rewards = np.asarray(
                info["counterfactual_branch_rewards"], dtype=np.float32
            )
            counterfactual_costs = np.asarray(
                info["counterfactual_branch_costs"], dtype=np.float32
            )
            delta_rewards = np.asarray(info["branch_delta_rewards"], dtype=np.float32)
            delta_costs = np.asarray(info["branch_delta_costs"], dtype=np.float32)
            branch_actual_rewards.append(
                np.full(4, float(reward), dtype=np.float32)
            )
            branch_counterfactual_rewards.append(counterfactual_rewards)
            branch_delta_rewards.append(delta_rewards)
            branch_actual_costs.append(
                np.full(4, float(info["constraint_cost"]), dtype=np.float32)
            )
            branch_counterfactual_costs.append(counterfactual_costs)
            branch_delta_costs.append(delta_costs)
            branch_rewards.append(
                delta_rewards
                if uses_counterfactual_reward(model.branch_credit_mode)
                else actual_branch_rewards
            )
            branch_costs.append(
                delta_costs
                if uses_counterfactual_cost(model.branch_credit_mode)
                else actual_branch_costs
            )
            counterfactual_weight_distances.append(
                np.asarray(info["counterfactual_weight_l1_distances"], dtype=np.float32)
            )
            counterfactual_turnover_differences.append(
                np.asarray(info["counterfactual_turnover_differences"], dtype=np.float32)
            )
            counterfactual_drawdown_differences.append(
                np.asarray(info["counterfactual_drawdown_differences"], dtype=np.float32)
            )
            counterfactual_zero_effects.append(
                np.asarray(info["counterfactual_zero_effects"], dtype=np.float32)
            )
            counterfactual_nonfinite_count += int(
                info["counterfactual_nonfinite_count"]
            )
            counterfactual_mapping_failure_count += int(
                info["counterfactual_mapping_failure_count"]
            )
            if branch_train_mask is None:
                branch_train_mask = np.asarray(
                    info["branch_train_mask"], dtype=np.float32)
            branch_z_values.append(np.asarray(info["branch_z_values"], dtype=np.float32))
            branch_turnovers.append(np.asarray(info["branch_turnovers"], dtype=np.float32))
            branch_transaction_costs.append(
                np.asarray(info["branch_transaction_costs"], dtype=np.float32)
            )
            branch_max_drawdowns.append(
                np.asarray(info["branch_max_drawdowns"], dtype=np.float32)
            )
            current_drawdowns.append(float(info["current_drawdown"]))
            max_drawdowns.append(float(info["max_drawdown"]))
            benchmark_current_drawdowns.append(float(info["benchmark_current_drawdown"]))
            benchmark_max_drawdowns.append(float(info["benchmark_max_drawdown"]))
            drawdown_benchmark_mode = str(info["drawdown_benchmark_mode"])
            effective_drawdown_budgets.append(float(info["effective_drawdown_budget"]))
            if alpha_budget_ratio is not None:
                if drawdown_cost_scale is None:
                    raise ValueError("drawdown_cost_scale is required with alpha_budget_ratio.")
                step_alpha_target = float(
                    (alpha_budget_ratio * float(info["effective_drawdown_budget"])) ** 2
                    / max(drawdown_cost_scale, 1e-12)
                )
            elif fixed_alpha_target is not None:
                step_alpha_target = float(fixed_alpha_target)
            else:
                step_alpha_target = 0.0
            alpha_targets.append(step_alpha_target)
            drawdown_gaps.append(float(info["drawdown_gap"]))
            drawdown_violations.append(float(info["drawdown_violation"]))
            drawdown_constraint_costs.append(float(info["drawdown_constraint_cost"]))
            allocation_constraint_1_violation_costs.append(
                float(info["allocation_constraint_1_violation_cost"])
            )
            allocation_constraint_2_violation_costs.append(
                float(info["allocation_constraint_2_violation_cost"])
            )
            allocation_constraint_raw_costs.append(float(info["allocation_constraint_raw_cost"]))
            allocation_constraint_costs.append(float(info["allocation_constraint_cost"]))
            allocation_drawdown_constraint_costs.append(
                float(info["allocation_drawdown_constraint_cost"])
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
        episode_alpha_target += step_alpha_target
        episode_turnover += float(info["turnover"])
        episode_steps += 1

        obs = next_obs
        if done:
            episode_metrics.append(
                {
                    "episode_return": episode_reward,
                    "episode_relative_wealth_vs_baseline": float(
                        info["portfolio_value"]
                        / max(float(info["benchmark_portfolio_value"]), 1e-12)
                        - 1.0
                    ),
                    "episode_cost": episode_cost / max(episode_steps, 1),
                    "episode_alpha_target": episode_alpha_target
                    / max(episode_steps, 1),
                    "episode_constraint_gap": (
                        episode_cost - episode_alpha_target
                    )
                    / max(episode_steps, 1),
                    "episode_turnover": episode_turnover / max(episode_steps, 1),
                }
            )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_alpha_target = 0.0
            episode_turnover = 0.0
            episode_steps = 0

    with profile_section(profiler, "policy_action_forward"):
        next_context_tensor = None
        if uses_counterfactual_context(model.branch_credit_mode):
            next_context_tensor = torch.as_tensor(
                env.counterfactual_critic_context(),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        next_value_r, next_value_c, next_branch_value_r, next_branch_value_c = (
            model.value_with_branches(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                counterfactual_context=next_context_tensor,
            )
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
        branch_log_probs_tensor = torch.as_tensor(
            np.asarray(branch_log_probs), dtype=torch.float32, device=device
        )
        branch_entropies_tensor = torch.as_tensor(
            np.asarray(branch_entropies), dtype=torch.float32, device=device
        )
        branch_rewards_tensor = torch.as_tensor(
            np.asarray(branch_rewards), dtype=torch.float32, device=device
        )
        branch_costs_tensor = torch.as_tensor(
            np.asarray(branch_costs), dtype=torch.float32, device=device
        )
        branch_z_values_tensor = torch.as_tensor(
            np.asarray(branch_z_values), dtype=torch.float32, device=device
        )
        branch_reward_values_tensor = torch.as_tensor(
            np.asarray(branch_reward_values), dtype=torch.float32, device=device
        )
        branch_cost_values_tensor = torch.as_tensor(
            np.asarray(branch_cost_values), dtype=torch.float32, device=device
        )
        branch_critic_contexts_tensor = torch.as_tensor(
            np.asarray(branch_critic_contexts), dtype=torch.float32, device=device
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
    if model.branch_credit_mode != "global":
        with profile_section(profiler, "reward_gae"):
            branch_reward_advantages, branch_reward_returns = compute_gae(
                branch_rewards_tensor,
                branch_reward_values_tensor,
                dones_tensor,
                next_branch_value_r.squeeze(0).detach(),
                optimization.gamma,
                optimization.gae_lambda,
            )
        if model.branch_credit_mode == "standalone" or uses_counterfactual_cost(
            model.branch_credit_mode
        ):
            with profile_section(profiler, "cost_gae"):
                branch_cost_advantages, branch_cost_returns = compute_gae(
                    branch_costs_tensor,
                    branch_cost_values_tensor,
                    dones_tensor,
                    next_branch_value_c.squeeze(0).detach(),
                    optimization.gamma,
                    optimization.gae_lambda,
                )
        else:
            branch_cost_advantages = cost_advantages.unsqueeze(-1).expand_as(
                branch_rewards_tensor
            )
            branch_cost_returns = cost_returns.unsqueeze(-1).expand_as(
                branch_rewards_tensor
            )
    else:
        branch_reward_advantages = torch.zeros_like(branch_rewards_tensor)
        branch_cost_advantages = torch.zeros_like(branch_costs_tensor)
        branch_reward_returns = torch.zeros_like(branch_rewards_tensor)
        branch_cost_returns = torch.zeros_like(branch_costs_tensor)

    episode_constraint_gaps = tuple(
        float(metric["episode_constraint_gap"]) for metric in episode_metrics
    )
    episode_feasible_rate = (
        float(np.mean(np.asarray(episode_constraint_gaps) <= 0.0))
        if episode_constraint_gaps
        else 0.0
    )
    episode_gap_p80 = (
        float(np.quantile(episode_constraint_gaps, 0.80, method="higher"))
        if episode_constraint_gaps
        else 0.0
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
        "batch_completed_episode_count": len(episode_constraint_gaps),
        "batch_episode_constraint_gap_mean": float(
            np.mean(episode_constraint_gaps)
        )
        if episode_constraint_gaps
        else 0.0,
        "batch_episode_constraint_gap_p80": episode_gap_p80,
        "batch_episode_constraint_gap_max": float(max(episode_constraint_gaps))
        if episode_constraint_gaps
        else 0.0,
        "batch_episode_feasible_rate": episode_feasible_rate,
        "batch_drawdown_gap_mean": float(np.mean(drawdown_gaps)),
        "batch_drawdown_violation_mean": float(np.mean(drawdown_violations)),
        "batch_drawdown_constraint_cost_mean": float(np.mean(drawdown_constraint_costs)),
        "allocation_max_violation": float(
            max(
                max(allocation_constraint_1_violation_costs, default=0.0),
                max(allocation_constraint_2_violation_costs, default=0.0),
            )
        ),
        "allocation_feasible": int(
            max(
                max(allocation_constraint_1_violation_costs, default=0.0),
                max(allocation_constraint_2_violation_costs, default=0.0),
            )
            <= 1e-10
        ),
        "batch_allocation_constraint_1_violation_cost_mean": float(
            np.mean(allocation_constraint_1_violation_costs)
        ),
        "batch_allocation_constraint_2_violation_cost_mean": float(
            np.mean(allocation_constraint_2_violation_costs)
        ),
        "batch_allocation_constraint_raw_cost_mean": float(
            np.mean(allocation_constraint_raw_costs)
        ),
        "batch_allocation_constraint_cost_mean": float(
            np.mean(allocation_constraint_costs)
        ),
        "batch_allocation_drawdown_constraint_cost_mean": float(
            np.mean(allocation_drawdown_constraint_costs)
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
        "episode_relative_wealth_vs_baseline_mean": _flatten_metrics(
            episode_metrics, "episode_relative_wealth_vs_baseline"
        ),
        "episode_cost_mean": _flatten_metrics(episode_metrics, "episode_cost"),
        **correction.metrics,
    }
    info_summary["batch_reward_advantage_std"] = float(
        reward_advantages.std(unbiased=False).item()
    )
    info_summary["batch_cost_advantage_std"] = float(
        cost_advantages.std(unbiased=False).item()
    )
    for branch_index in range(branch_rewards_tensor.shape[1]):
        number = branch_index + 1
        info_summary[f"batch_branch_{number}_reward_mean"] = float(
            branch_rewards_tensor[:, branch_index].mean().item()
        )
        info_summary[f"batch_branch_{number}_training_active"] = int(
            branch_train_mask is not None
            and bool(branch_train_mask[branch_index])
        )
        info_summary[f"batch_branch_{number}_transaction_cost_mean"] = float(
            np.mean(np.asarray(branch_transaction_costs)[:, branch_index])
        )
        info_summary[f"batch_branch_{number}_drawdown_cost_mean"] = float(
            branch_costs_tensor[:, branch_index].mean().item()
        )
        info_summary[f"batch_branch_{number}_max_drawdown_mean"] = float(
            np.mean(np.asarray(branch_max_drawdowns)[:, branch_index])
        )
        info_summary[f"batch_branch_{number}_z_mean"] = float(
            branch_z_values_tensor[:, branch_index].mean().item()
        )
        info_summary[f"batch_branch_{number}_reward_advantage_std"] = float(
            branch_reward_advantages[:, branch_index].std(unbiased=False).item()
        )
        info_summary[f"batch_branch_{number}_cost_advantage_std"] = float(
            branch_cost_advantages[:, branch_index].std(unbiased=False).item()
        )
        if uses_counterfactual_context(model.branch_credit_mode):
            actual_rewards_array = np.asarray(branch_actual_rewards)[:, branch_index]
            counterfactual_rewards_array = np.asarray(
                branch_counterfactual_rewards
            )[:, branch_index]
            delta_rewards_array = np.asarray(branch_delta_rewards)[:, branch_index]
            actual_costs_array = np.asarray(branch_actual_costs)[:, branch_index]
            counterfactual_costs_array = np.asarray(
                branch_counterfactual_costs
            )[:, branch_index]
            delta_costs_array = np.asarray(branch_delta_costs)[:, branch_index]
            info_summary[f"branch_actual_reward_mean_{number}"] = float(
                np.mean(actual_rewards_array)
            )
            info_summary[f"branch_counterfactual_reward_mean_{number}"] = float(
                np.mean(counterfactual_rewards_array)
            )
            info_summary[f"branch_delta_reward_mean_{number}"] = float(
                np.mean(delta_rewards_array)
            )
            info_summary[f"branch_delta_reward_std_{number}"] = float(
                np.std(delta_rewards_array)
            )
            info_summary[f"branch_actual_cost_mean_{number}"] = float(
                np.mean(actual_costs_array)
            )
            info_summary[f"branch_counterfactual_cost_mean_{number}"] = float(
                np.mean(counterfactual_costs_array)
            )
            info_summary[f"branch_delta_cost_mean_{number}"] = float(
                np.mean(delta_costs_array)
            )
            info_summary[f"branch_delta_cost_std_{number}"] = float(
                np.std(delta_costs_array)
            )
            info_summary[f"counterfactual_weight_l1_distance_mean_{number}"] = float(
                np.mean(np.asarray(counterfactual_weight_distances)[:, branch_index])
            )
            info_summary[f"counterfactual_turnover_difference_mean_{number}"] = float(
                np.mean(
                    np.asarray(counterfactual_turnover_differences)[:, branch_index]
                )
            )
            info_summary[f"counterfactual_drawdown_difference_mean_{number}"] = float(
                np.mean(
                    np.asarray(counterfactual_drawdown_differences)[:, branch_index]
                )
            )
            info_summary[f"counterfactual_zero_effect_rate_{number}"] = float(
                np.mean(np.asarray(counterfactual_zero_effects)[:, branch_index])
            )
    info_summary["counterfactual_nonfinite_count"] = int(
        counterfactual_nonfinite_count
    )
    info_summary["counterfactual_mapping_failure_count"] = int(
        counterfactual_mapping_failure_count
    )
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
        branch_log_probs=branch_log_probs_tensor,
        branch_entropies=branch_entropies_tensor,
        branch_rewards=branch_rewards_tensor,
        branch_costs=branch_costs_tensor,
        branch_z_values=branch_z_values_tensor,
        branch_reward_values=branch_reward_values_tensor,
        branch_cost_values=branch_cost_values_tensor,
        branch_reward_returns=branch_reward_returns.detach(),
        branch_cost_returns=branch_cost_returns.detach(),
        branch_reward_advantages=branch_reward_advantages.detach(),
        branch_cost_advantages=branch_cost_advantages.detach(),
        branch_critic_contexts=branch_critic_contexts_tensor,
        info_summary=info_summary,
        episode_constraint_gaps=episode_constraint_gaps,
    )
