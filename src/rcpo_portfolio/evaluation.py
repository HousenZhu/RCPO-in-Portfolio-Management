from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import (
    BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION,
    ProjectConfig,
    sync_rcpo_constraint_settings,
)
from .devices import resolve_device
from .env import PortfolioEnv
from .market import generate_market_splits
from .models import ActorCritic


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    first_episode: dict[str, np.ndarray]
    episode_returns: list[np.ndarray]
    equal_weight_first_episode_returns: np.ndarray
    equal_weight_episode_returns: list[np.ndarray]


def compute_drawdown(returns: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(wealth)
    return (running_peak - wealth) / np.maximum(running_peak, 1e-12)


def summarize_returns(returns: np.ndarray, turnover: np.ndarray) -> dict[str, float]:
    log_growth = np.log1p(np.clip(returns, -0.999999, None))
    annualized_return = float(np.exp(log_growth.mean() * 252.0) - 1.0)
    annualized_volatility = float(np.std(returns) * np.sqrt(252.0))
    max_drawdown = float(np.max(compute_drawdown(returns)))
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "average_turnover": float(np.mean(turnover)),
    }


def select_start_indices(env: PortfolioEnv, episodes: int) -> np.ndarray:
    starts = env.available_start_indices()
    if episodes >= len(starts):
        return starts
    indices = np.linspace(0, len(starts) - 1, num=episodes, dtype=int)
    return starts[indices]


def _equal_weight_logits(env: PortfolioEnv) -> np.ndarray:
    return np.zeros(env.action_space.shape[0], dtype=np.float32)


def _rollout_returns(
    env: PortfolioEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    start_index: int,
) -> np.ndarray:
    obs, _ = env.reset(options={"start_index": int(start_index)})
    net_returns: list[float] = []
    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        del reward
        net_returns.append(float(info["net_return"]))
        if terminated or truncated:
            break
    return np.asarray(net_returns, dtype=np.float32)


def evaluate_policy(
    env: PortfolioEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    alpha: float | None,
    alpha_budget_ratio: float | None,
    split_name: str,
) -> EvaluationResult:
    episode_summaries: list[dict[str, float]] = []
    episode_return_paths: list[np.ndarray] = []
    equal_weight_return_paths: list[np.ndarray] = []
    first_episode: dict[str, np.ndarray] | None = None
    first_start_index: int | None = None
    violation_count = 0
    start_indices = select_start_indices(env, episodes)
    for start_index in start_indices:
        obs, _ = env.reset(options={"start_index": int(start_index)})
        net_returns: list[float] = []
        constraint_costs: list[float] = []
        current_drawdowns: list[float] = []
        max_drawdowns: list[float] = []
        benchmark_current_drawdowns: list[float] = []
        benchmark_max_drawdowns: list[float] = []
        effective_drawdown_budgets: list[float] = []
        alpha_targets: list[float] = []
        drawdown_gaps: list[float] = []
        drawdown_violations: list[float] = []
        drawdown_constraint_costs: list[float] = []
        group_a_min_violation_costs: list[float] = []
        group_b_max_violation_costs: list[float] = []
        group_a_weights: list[float] = []
        group_b_weights: list[float] = []
        concentrations: list[float] = []
        excess_concentration_costs: list[float] = []
        diversification_costs: list[float] = []
        turnover: list[float] = []
        weights: list[np.ndarray] = []
        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            del reward
            net_returns.append(float(info["net_return"]))
            constraint_costs.append(float(info["constraint_cost"]))
            current_drawdowns.append(float(info["current_drawdown"]))
            max_drawdowns.append(float(info["max_drawdown"]))
            benchmark_current_drawdowns.append(float(info["benchmark_current_drawdown"]))
            benchmark_max_drawdowns.append(float(info["benchmark_max_drawdown"]))
            effective_drawdown_budgets.append(float(info["effective_drawdown_budget"]))
            if alpha_budget_ratio is not None:
                alpha_targets.append(
                    float(
                        (alpha_budget_ratio * float(info["effective_drawdown_budget"])) ** 2
                        / max(float(env.config.drawdown_cost_scale), 1e-12)
                    )
                )
            drawdown_gaps.append(float(info["drawdown_gap"]))
            drawdown_violations.append(float(info["drawdown_violation"]))
            drawdown_constraint_costs.append(float(info["drawdown_constraint_cost"]))
            group_a_min_violation_costs.append(float(info["group_a_min_violation_cost"]))
            group_b_max_violation_costs.append(float(info["group_b_max_violation_cost"]))
            group_a_weights.append(float(info["group_a_weight"]))
            group_b_weights.append(float(info["group_b_weight"]))
            concentrations.append(float(info["concentration"]))
            excess_concentration_costs.append(float(info["excess_concentration_cost"]))
            diversification_costs.append(float(info["diversification_cost"]))
            turnover.append(float(info["turnover"]))
            weights.append(np.asarray(info["weights"], dtype=np.float32))
            if terminated or truncated:
                break
        episode_returns = np.asarray(net_returns, dtype=np.float32)
        episode_constraint = np.asarray(constraint_costs, dtype=np.float32)
        episode_turnover = np.asarray(turnover, dtype=np.float32)
        episode_current_drawdown = np.asarray(current_drawdowns, dtype=np.float32)
        episode_max_drawdown = np.asarray(max_drawdowns, dtype=np.float32)
        episode_benchmark_current_drawdown = np.asarray(
            benchmark_current_drawdowns,
            dtype=np.float32,
        )
        episode_benchmark_max_drawdown = np.asarray(
            benchmark_max_drawdowns,
            dtype=np.float32,
        )
        episode_effective_drawdown_budget = np.asarray(
            effective_drawdown_budgets,
            dtype=np.float32,
        )
        episode_alpha_target = np.asarray(alpha_targets, dtype=np.float32)
        episode_drawdown_gap = np.asarray(drawdown_gaps, dtype=np.float32)
        episode_drawdown_violation = np.asarray(drawdown_violations, dtype=np.float32)
        episode_drawdown_constraint_cost = np.asarray(
            drawdown_constraint_costs,
            dtype=np.float32,
        )
        episode_summary = summarize_returns(episode_returns, episode_turnover)
        episode_summary["episode_average_constraint_cost"] = float(np.mean(episode_constraint))
        episode_summary["average_constraint_cost"] = float(np.mean(episode_constraint))
        episode_summary["average_current_drawdown"] = float(np.mean(episode_current_drawdown))
        episode_summary["average_step_max_drawdown"] = float(np.mean(episode_max_drawdown))
        episode_summary["average_benchmark_current_drawdown"] = float(
            np.mean(episode_benchmark_current_drawdown)
        )
        episode_summary["average_step_benchmark_max_drawdown"] = float(
            np.mean(episode_benchmark_max_drawdown)
        )
        episode_summary["benchmark_max_drawdown"] = float(
            episode_benchmark_max_drawdown[-1]
        )
        episode_summary["effective_drawdown_budget"] = float(
            episode_effective_drawdown_budget[-1]
        )
        episode_summary["average_effective_drawdown_budget"] = float(
            np.mean(episode_effective_drawdown_budget)
        )
        episode_summary["average_alpha_target"] = float(
            np.mean(episode_alpha_target)
        ) if len(episode_alpha_target) > 0 else (float(alpha) if alpha is not None else 0.0)
        episode_summary["average_drawdown_gap"] = float(np.mean(episode_drawdown_gap))
        episode_summary["average_drawdown_violation"] = float(np.mean(episode_drawdown_violation))
        episode_summary["average_drawdown_constraint_cost"] = float(
            np.mean(episode_drawdown_constraint_cost)
        )
        episode_summary["average_group_a_min_violation_cost"] = float(np.mean(group_a_min_violation_costs))
        episode_summary["average_group_b_max_violation_cost"] = float(np.mean(group_b_max_violation_costs))
        episode_summary["average_group_a_weight"] = float(np.mean(group_a_weights))
        episode_summary["average_group_b_weight"] = float(np.mean(group_b_weights))
        episode_summary["average_concentration"] = float(np.mean(concentrations))
        episode_summary["average_excess_concentration_cost"] = float(
            np.mean(excess_concentration_costs)
        )
        episode_summary["average_diversification_cost"] = float(
            np.mean(diversification_costs)
        )
        episode_summaries.append(episode_summary)
        episode_return_paths.append(episode_returns)
        equal_weight_return_paths.append(
            _rollout_returns(env, lambda _obs: _equal_weight_logits(env), int(start_index))
        )
        episode_alpha_threshold = (
            episode_summary["average_alpha_target"]
            if alpha_budget_ratio is not None
            else alpha
        )
        if (
            episode_alpha_threshold is not None
            and episode_summary["episode_average_constraint_cost"] > episode_alpha_threshold
        ):
            violation_count += 1
        if first_episode is None:
            first_start_index = int(start_index)
            first_episode = {
                "net_returns": episode_returns,
                "constraint_costs": episode_constraint,
                "current_drawdowns": episode_current_drawdown,
                "max_drawdowns": episode_max_drawdown,
                "benchmark_current_drawdowns": episode_benchmark_current_drawdown,
                "benchmark_max_drawdowns": episode_benchmark_max_drawdown,
                "effective_drawdown_budgets": episode_effective_drawdown_budget,
                "alpha_targets": episode_alpha_target,
                "drawdown_gaps": episode_drawdown_gap,
                "drawdown_violations": episode_drawdown_violation,
                "drawdown_constraint_costs": episode_drawdown_constraint_cost,
                "turnover": episode_turnover,
                "group_a_weights": np.asarray(group_a_weights, dtype=np.float32),
                "group_b_weights": np.asarray(group_b_weights, dtype=np.float32),
                "concentrations": np.asarray(concentrations, dtype=np.float32),
                "excess_concentration_costs": np.asarray(
                    excess_concentration_costs,
                    dtype=np.float32,
                ),
                "diversification_costs": np.asarray(diversification_costs, dtype=np.float32),
                "weights": np.asarray(weights, dtype=np.float32),
            }
    if first_episode is None:
        raise RuntimeError("Evaluation did not produce any episodes.")
    if first_start_index is None:
        raise RuntimeError("Evaluation did not record a first start index.")
    aggregate = {
        metric: float(np.mean([episode[metric] for episode in episode_summaries]))
        for metric in episode_summaries[0]
    }
    aggregate["episodes"] = len(episode_summaries)
    aggregate["constraint_violation_rate"] = (
        float(violation_count / len(episode_summaries))
        if (alpha is not None or alpha_budget_ratio is not None)
        else 0.0
    )
    aggregate["split"] = split_name
    return EvaluationResult(
        summary=aggregate,
        first_episode=first_episode,
        episode_returns=episode_return_paths,
        equal_weight_first_episode_returns=_rollout_returns(
            env, lambda _obs: _equal_weight_logits(env), first_start_index
        ),
        equal_weight_episode_returns=equal_weight_return_paths,
    )


def save_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: str | Path,
    split_name: str,
    rolling_window: int,
    lambda_history: list[float] | None = None,
    mean_episode_returns: list[np.ndarray] | None = None,
    equal_weight_mean_episode_returns: list[np.ndarray] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / f"summary_{split_name}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2)

    returns = result.first_episode["net_returns"]
    turnover = result.first_episode["turnover"]
    weights = result.first_episode["weights"]
    drawdown = compute_drawdown(returns)
    drawdown_constraint_costs = result.first_episode["drawdown_constraint_costs"]
    cumulative_return = np.cumprod(1.0 + returns) - 1.0
    equal_weight_cumulative_return = (
        np.cumprod(1.0 + result.equal_weight_first_episode_returns) - 1.0
    )
    plt.figure(figsize=(8, 4))
    plt.plot(cumulative_return, label="Model")
    plt.plot(equal_weight_cumulative_return, label="Equal Weight", linestyle="--")
    plt.title(f"Cumulative Return ({split_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"cumulative_return_{split_name}.png")
    plt.close()

    plots = [
        ("turnover", turnover, "Turnover"),
        ("drawdown", drawdown, "Drawdown"),
        (
            "drawdown_constraint_cost",
            drawdown_constraint_costs,
            "Drawdown Constraint Cost",
        ),
    ]
    for file_stem, series, title in plots:
        plt.figure(figsize=(8, 4))
        plt.plot(series)
        plt.title(f"{title} ({split_name})")
        plt.tight_layout()
        plt.savefig(output_path / f"{file_stem}_{split_name}.png")
        plt.close()

    model_mean_paths = mean_episode_returns or result.episode_returns
    equal_weight_mean_paths = equal_weight_mean_episode_returns or result.equal_weight_episode_returns
    if model_mean_paths and equal_weight_mean_paths:
        min_length = min(len(path) for path in model_mean_paths)
        cumulative_paths = np.asarray(
            [
                np.cumprod(1.0 + path[:min_length]) - 1.0
                for path in model_mean_paths
            ],
            dtype=np.float32,
        )
        mean_cumulative_return = cumulative_paths.mean(axis=0)
        equal_weight_min_length = min(len(path) for path in equal_weight_mean_paths)
        equal_weight_cumulative_paths = np.asarray(
            [
                np.cumprod(1.0 + path[:equal_weight_min_length]) - 1.0
                for path in equal_weight_mean_paths
            ],
            dtype=np.float32,
        )
        common_length = min(len(mean_cumulative_return), equal_weight_cumulative_paths.shape[1])
        mean_cumulative_return = mean_cumulative_return[:common_length]
        equal_weight_mean_cumulative_return = equal_weight_cumulative_paths[:, :common_length].mean(axis=0)
        plt.figure(figsize=(8, 4))
        for path in cumulative_paths[:, :common_length]:
            plt.plot(path, color="#9ecae1", alpha=0.35, linewidth=1.0)
        plt.plot(mean_cumulative_return, color="#08519c", linewidth=2.4, label="Mean")
        plt.plot(
            equal_weight_mean_cumulative_return,
            color="#d94801",
            linewidth=2.0,
            linestyle="--",
            label="Equal Weight Mean",
        )
        plt.title(f"Mean Cumulative Return ({split_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"mean_cumulative_return_{split_name}.png")
        plt.close()

    save_group_weights_artifact(result, output_path, split_name)

    if lambda_history is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(lambda_history)
        plt.title("Lambda Trajectory")
        plt.tight_layout()
        plt.savefig(output_path / "lambda_trajectory.png")
        plt.close()


def save_group_weights_artifact(
    result: EvaluationResult,
    output_dir: str | Path,
    split_name: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    weights = result.first_episode["weights"]
    plt.figure(figsize=(10, 5))
    asset_labels = ["Cash", *[f"Asset {index}" for index in range(1, weights.shape[1])]]
    for asset_index, label in enumerate(asset_labels):
        plt.plot(weights[:, asset_index], label=label, linewidth=1.2, alpha=0.85)
    plt.title(f"Portfolio Weights ({split_name})")
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_path / f"group_weights_{split_name}.png")
    plt.close()


def save_training_progress_artifacts(
    metrics_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    if not metrics_rows:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    updates = np.asarray([row["update"] + 1 for row in metrics_rows], dtype=np.int32)
    rollout_returns = np.asarray([row["episode_return_mean"] for row in metrics_rows], dtype=np.float32)
    evaluation_prefix = (
        "validation" if "validation_annualized_return" in metrics_rows[0] else "test"
    )
    evaluation_returns = np.asarray(
        [row[f"{evaluation_prefix}_annualized_return"] for row in metrics_rows],
        dtype=np.float32,
    )
    rollout_violation = np.asarray(
        [
            (
                row["alpha"] is not None
                and row["batch_constraint_cost_mean"] > row["alpha"]
            )
            for row in metrics_rows
        ],
        dtype=bool,
    )
    evaluation_violation = np.asarray(
        [
            (
                row.get(f"{evaluation_prefix}_alpha_target", row["alpha"]) is not None
                and row[f"{evaluation_prefix}_constraint_cost"]
                > row.get(f"{evaluation_prefix}_alpha_target", row["alpha"])
            )
            for row in metrics_rows
        ],
        dtype=bool,
    )

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(updates, rollout_returns, label="Rollout Return", color="#1f77b4")
    axis.plot(
        updates,
        evaluation_returns,
        label=f"{evaluation_prefix.title()} Annualized Return",
        color="#ff7f0e",
    )

    rollout_label_used = False
    for index, is_violating in enumerate(rollout_violation):
        if is_violating:
            axis.axvspan(
                updates[index] - 0.5,
                updates[index] + 0.5,
                color="#1f77b4",
                alpha=0.10,
                label="Rollout constraint violation" if not rollout_label_used else None,
            )
            rollout_label_used = True
    evaluation_label_used = False
    for index, is_violating in enumerate(evaluation_violation):
        if is_violating:
            axis.axvspan(
                updates[index] - 0.5,
                updates[index] + 0.5,
                color="#d62728",
                alpha=0.12,
                label=f"{evaluation_prefix.title()} constraint violation"
                if not evaluation_label_used
                else None,
            )
            evaluation_label_used = True

    constraint_mode = str(metrics_rows[0].get("constraint_mode", "selected"))
    axis.set_title(
        f"Training Return With {constraint_mode.replace('_', ' ').title()} Constraint Violations"
    )
    axis.set_xlabel("Update")
    axis.set_ylabel("Return")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path / "training_return.png")
    plt.close(fig)

    rollout_turnover = np.asarray(
        [row.get("batch_turnover_mean", np.nan) for row in metrics_rows],
        dtype=np.float32,
    )
    validation_turnover = np.asarray(
        [row.get(f"{evaluation_prefix}_turnover", np.nan) for row in metrics_rows],
        dtype=np.float32,
    )
    turnover_caps = [
        float(row["turnover_cap"])
        for row in metrics_rows
        if row.get("turnover_cap") is not None
    ]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(updates, rollout_turnover, label="Rollout Turnover", color="#2ca02c")
    if np.isfinite(validation_turnover).any():
        axis.plot(
            updates,
            validation_turnover,
            label=f"{evaluation_prefix.title()} Turnover",
            color="#9467bd",
        )
    if turnover_caps:
        axis.axhline(
            turnover_caps[-1],
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label="Turnover Cap",
        )
    axis.set_title("Training Turnover")
    axis.set_xlabel("Update")
    axis.set_ylabel("Average Turnover")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path / "training_turnover.png")
    plt.close(fig)

    reward_modes = {str(row.get("reward_correction_mode", "none")) for row in metrics_rows}
    if reward_modes != {"none"}:
        observed_rewards = np.asarray(
            [row.get("observed_reward_mean", np.nan) for row in metrics_rows],
            dtype=np.float32,
        )
        corrected_rewards = np.asarray(
            [row.get("corrected_reward_mean", np.nan) for row in metrics_rows],
            dtype=np.float32,
        )
        correction_abs = np.asarray(
            [row.get("reward_correction_delta_abs_mean", np.nan) for row in metrics_rows],
            dtype=np.float32,
        )
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(updates, observed_rewards, label="Observed Reward", color="#1f77b4")
        axis.plot(updates, corrected_rewards, label="Corrected Reward", color="#ff7f0e")
        axis.plot(
            updates,
            correction_abs,
            label="Mean Abs Correction",
            color="#2ca02c",
            alpha=0.85,
        )
        axis.set_title("Training Reward Correction")
        axis.set_xlabel("Update")
        axis.set_ylabel("Reward")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_path / "training_reward_correction.png")
        plt.close(fig)

    selected_bins = np.asarray(
        [row.get("gdrc_selected_bins", 0) for row in metrics_rows],
        dtype=np.float32,
    )
    if np.any(selected_bins > 0):
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.step(updates, selected_bins, where="mid", color="#6a3d9a")
        axis.set_title("GDRC Selected Reward Bins")
        axis.set_xlabel("Update")
        axis.set_ylabel("Selected Bins")
        fig.tight_layout()
        fig.savefig(output_path / "gdrc_selected_bins.png")
        plt.close(fig)


def load_checkpoint_for_evaluation(
    run_dir: str | Path,
    checkpoint_name: str = "checkpoint_best.pt",
) -> tuple[ProjectConfig, dict[str, Any], ActorCritic, dict[str, PortfolioEnv]]:
    run_path = Path(run_dir)
    from .config import load_config

    config = load_config(run_path / "config_snapshot.yaml")
    sync_rcpo_constraint_settings(config)
    device = resolve_device(config.runtime.device)
    checkpoint = torch.load(run_path / checkpoint_name, map_location=device)
    if checkpoint.get("algo") == "rcpo":
        checkpoint_constraint_mode = checkpoint.get("constraint_mode")
        if checkpoint_constraint_mode != config.rcpo.constraint_mode:
            raise ValueError(
                f"Checkpoint constraint mode {checkpoint_constraint_mode!r} does not match "
                f"config_snapshot {config.rcpo.constraint_mode!r}."
            )
        checkpoint_semantics = checkpoint.get("constraint_semantics")
        if checkpoint_semantics != BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION:
            raise ValueError(
                "RCPO checkpoint uses incompatible drawdown constraint semantics. "
                "Legacy fixed-budget drawdown checkpoints are not supported."
            )
        if not np.isclose(
            float(checkpoint.get("drawdown_budget_floor", np.nan)),
            float(config.environment.drawdown_budget_floor),
        ):
            raise ValueError("Checkpoint drawdown_budget_floor does not match config_snapshot.")
        if not np.isclose(
            float(checkpoint.get("benchmark_drawdown_margin", np.nan)),
            float(config.environment.benchmark_drawdown_margin),
        ):
            raise ValueError(
                "Checkpoint benchmark_drawdown_margin does not match config_snapshot."
            )
        if not np.isclose(
            float(checkpoint.get("drawdown_cost_scale", np.nan)),
            float(config.environment.drawdown_cost_scale),
        ):
            raise ValueError("Checkpoint drawdown_cost_scale does not match config_snapshot.")
    market_splits = generate_market_splits(config.market, int(checkpoint["seed"]))
    environments = {
        split_name: PortfolioEnv(config.environment, market, config.market, seed=int(checkpoint["seed"]))
        for split_name, market in market_splits.items()
    }
    model = ActorCritic(
        obs_dim=environments["train"].observation_space.shape[0],
        action_dim=environments["train"].action_space.shape[0],
        config=config.network,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "algo": checkpoint["algo"],
        "seed": int(checkpoint["seed"]),
        "alpha": float(checkpoint["alpha"]) if checkpoint["alpha"] is not None else None,
        "alpha_budget_ratio": float(
            checkpoint.get("alpha_budget_ratio", config.rcpo.alpha_budget_ratio)
        ),
        "constraint_mode": checkpoint.get("constraint_mode", config.rcpo.constraint_mode),
        "constraint_semantics": checkpoint.get(
            "constraint_semantics",
            BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION,
        ),
        "lambda_value": float(checkpoint["lambda_value"]),
        "device": str(device),
    }
    return config, metadata, model, environments
