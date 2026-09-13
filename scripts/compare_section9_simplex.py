from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.evaluation import (
    apply_signed_percent_axis,
    compute_drawdown,
    load_checkpoint_for_evaluation,
    relative_wealth_path,
)
from rcpo_portfolio.market import generate_continuation_splits


@dataclass(frozen=True)
class PolicySpec:
    label: str
    run_dir: Path
    checkpoint: str = "checkpoint_best_return.pt"


DEFAULT_POLICIES = [
    PolicySpec(
        label="PPO Dirichlet",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.1_ppo_unconstrained_none_20260620_144053"
        / "seed_0",
    ),
    PolicySpec(
        label="PPO Gaussian",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2_ppo_unconstrained_none_20260612_195706"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Dirichlet",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.1_rcpo_none_20260620_144057"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Gaussian",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.1_rcpo_none_20260620_144107"
        / "seed_0",
    ),
]


V22_POLICIES = [
    PolicySpec(
        label="PPO Gaussian v2.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.2_ppo_unconstrained_none_20260621_143156"
        / "seed_0",
    ),
    PolicySpec(
        label="PPO Dirichlet v2.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.2_ppo_unconstrained_none_20260621_162247"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Gaussian v2.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.2_rcpo_none_20260621_143215"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Dirichlet v2.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.2_rcpo_none_20260621_143223"
        / "seed_0",
    ),
]


V23_POLICIES = [
    PolicySpec(
        label="PPO Gaussian v2.3",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.3_ppo_unconstrained_none_20260707_204433"
        / "seed_0",
    ),
    PolicySpec(
        label="PPO Dirichlet v2.3",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.3_ppo_unconstrained_none_20260707_204606"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Gaussian v2.3",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.3_rcpo_none_20260707_204441"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Dirichlet v2.3",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.3_rcpo_none_20260707_204616"
        / "seed_0",
    ),
]


V24_POLICIES = [
    PolicySpec(
        label="PPO Gaussian v2.4",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.4_ppo_gaussian_ppo_unconstrained_none_20260718_162406"
        / "seed_0",
    ),
    PolicySpec(
        label="PPO Dirichlet v2.4",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.4_ppo_dirichlet_ppo_unconstrained_none_20260718_162418"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Gaussian v2.4",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.4_rcpo_gaussian_rcpo_none_20260718_162411"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Dirichlet v2.4",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.4_rcpo_dirichlet_rcpo_none_20260718_162422"
        / "seed_0",
    ),
]


V25_POLICIES = [
    PolicySpec(
        label="PPO Gaussian v2.5",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.5_ppo_gaussian_ppo_unconstrained_none_20260726_153052"
        / "seed_0",
    ),
    PolicySpec(
        label="PPO Dirichlet v2.5",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.5_ppo_dirichlet_ppo_unconstrained_none_20260726_153100"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Gaussian v2.5",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.5_rcpo_gaussian_rcpo_none_20260726_153056"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Dirichlet v2.5",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.5_rcpo_dirichlet_rcpo_none_20260726_153104"
        / "seed_0",
    ),
]


V26_POLICIES = [
    PolicySpec(
        label="PPO Gaussian v2.6",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_ppo_gaussian_ppo_unconstrained_none_20260802_231120"
        / "seed_0",
    ),
    PolicySpec(
        label="PPO Dirichlet v2.6",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_ppo_dirichlet_ppo_unconstrained_none_20260802_231122"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Gaussian v2.6",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_rcpo_gaussian_rcpo_none_20260802_231121"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Dirichlet v2.6",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_rcpo_dirichlet_rcpo_none_20260802_231123"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Allocation Penalty",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_rcpo_allocation_penalty_rcpo_none_20260802_220443"
        / "seed_0",
    ),
    PolicySpec(
        label="RCPO Allocation + Drawdown",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_rcpo_allocation_relative_drawdown_rcpo_none_20260802_220448"
        / "seed_0",
    ),
]


V32_POLICIES = [
    PolicySpec(
        label="CF Reward Gaussian v3.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v3.2_cf_reward_gaussian_rcpo_none_20260902_195405"
        / "seed_0",
        checkpoint="checkpoint_best_return.pt",
    ),
    PolicySpec(
        label="CF Cost Gaussian v3.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v3.2_cf_cost_gaussian_rcpo_none_20260902_195411"
        / "seed_0",
        checkpoint="checkpoint_best_return.pt",
    ),
    PolicySpec(
        label="CF Reward+Cost Gaussian v3.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v3.2_cf_reward_cost_gaussian_rcpo_none_20260902_195417"
        / "seed_0",
        checkpoint="checkpoint_best_return.pt",
    ),
    PolicySpec(
        label="CF Reward Dirichlet v3.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v3.2_cf_reward_dirichlet_rcpo_none_20260902_195424"
        / "seed_0",
        checkpoint="checkpoint_best_return.pt",
    ),
    PolicySpec(
        label="CF Cost Dirichlet v3.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v3.2_cf_cost_dirichlet_rcpo_none_20260902_195431"
        / "seed_0",
        checkpoint="checkpoint_best_return.pt",
    ),
    PolicySpec(
        label="CF Reward+Cost Dirichlet v3.2",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v3.2_cf_reward_cost_dirichlet_rcpo_none_20260902_195436"
        / "seed_0",
        checkpoint="checkpoint_best_return.pt",
    ),
]


V26_CONTROL_POLICIES = [
    PolicySpec(
        label="V2.6 Gaussian Control (Best Return)",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_rcpo_gaussian_rcpo_none_20260802_231121"
        / "seed_0",
        checkpoint="checkpoint_best_return_eval_snapshot.pt",
    ),
    PolicySpec(
        label="V2.6 Dirichlet Control (Best Return)",
        run_dir=PROJECT_ROOT
        / "runs"
        / "simplex_v2.6_rcpo_dirichlet_rcpo_none_20260802_231123"
        / "seed_0",
        checkpoint="checkpoint_best_return_eval_snapshot.pt",
    ),
]

ALLOCATION_PENALTY_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "rcpo_allocation_penalty_rcpo_none_20260630_141127"
    / "seed_0"
)

ALLOCATION_DRAWDOWN_PENALTY_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "rcpo_allocation_drawdown_penalty_rcpo_none_20260718_162434"
    / "seed_0"
)

POLICY_COLORS = {
    "ppo_gaussian": "#1f77b4",
    "ppo_dirichlet": "#ff7f0e",
    "rcpo_gaussian": "#2ca02c",
    "rcpo_dirichlet": "#d62728",
    "rcpo_allocation_penalty": "#9467bd",
    "rcpo_allocation_drawdown_penalty": "#8c564b",
    "constrained_neutral": "#111111",
    "cf_reward_gaussian": "#1f77b4",
    "cf_cost_gaussian": "#17becf",
    "cf_reward_cost_gaussian": "#2ca02c",
    "cf_reward_dirichlet": "#ff7f0e",
    "cf_cost_dirichlet": "#9467bd",
    "cf_reward_cost_dirichlet": "#d62728",
    "v26_gaussian": "#4c4c4c",
    "v26_dirichlet": "#bcbd22",
}


def policy_color(label: str) -> str:
    if label == "V2.6 Gaussian Control (Best Return)":
        return POLICY_COLORS["v26_gaussian"]
    if label == "V2.6 Dirichlet Control (Best Return)":
        return POLICY_COLORS["v26_dirichlet"]
    if label.startswith("CF Reward+Cost Gaussian"):
        return POLICY_COLORS["cf_reward_cost_gaussian"]
    if label.startswith("CF Cost Gaussian"):
        return POLICY_COLORS["cf_cost_gaussian"]
    if label.startswith("CF Reward Gaussian"):
        return POLICY_COLORS["cf_reward_gaussian"]
    if label.startswith("CF Reward+Cost Dirichlet"):
        return POLICY_COLORS["cf_reward_cost_dirichlet"]
    if label.startswith("CF Cost Dirichlet"):
        return POLICY_COLORS["cf_cost_dirichlet"]
    if label.startswith("CF Reward Dirichlet"):
        return POLICY_COLORS["cf_reward_dirichlet"]
    if label.startswith("PPO Gaussian"):
        return POLICY_COLORS["ppo_gaussian"]
    if label.startswith("PPO Dirichlet"):
        return POLICY_COLORS["ppo_dirichlet"]
    if label.startswith("RCPO Gaussian"):
        return POLICY_COLORS["rcpo_gaussian"]
    if label.startswith("RCPO Dirichlet"):
        return POLICY_COLORS["rcpo_dirichlet"]
    if label == "RCPO Allocation Penalty":
        return POLICY_COLORS["rcpo_allocation_penalty"]
    if label == "RCPO Allocation + Drawdown":
        return POLICY_COLORS["rcpo_allocation_drawdown_penalty"]
    return POLICY_COLORS["constrained_neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Section 9 shared comparison graphs for simplex policies."
    )
    parser.add_argument(
        "--policy-set",
        choices=["default", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3.2"],
        default="default",
        help="Which Section 9 simplex policy set to compare.",
    )
    parser.add_argument(
        "--include-allocation-penalty",
        action="store_true",
        help="Include the softmax RCPO allocation-penalty baseline as an extra policy.",
    )
    parser.add_argument(
        "--allocation-run-dir",
        default=str(ALLOCATION_PENALTY_RUN_DIR),
        help="Seed directory for the trained RCPO allocation-penalty run.",
    )
    parser.add_argument(
        "--include-allocation-drawdown-penalty",
        action="store_true",
        help="Include the softmax RCPO allocation-plus-drawdown baseline.",
    )
    parser.add_argument(
        "--allocation-drawdown-run-dir",
        default=str(ALLOCATION_DRAWDOWN_PENALTY_RUN_DIR),
        help="Seed directory for the trained RCPO allocation-plus-drawdown run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "section9_simplex_policy_comparison"),
        help="Folder where comparison PNG/CSV/JSON files are written.",
    )
    parser.add_argument(
        "--future-market-count",
        type=int,
        default=None,
        help="Override the configured number of shared continuation markets.",
    )
    parser.add_argument(
        "--future-seed-offset",
        type=int,
        default=None,
        help="Override the split-specific continuation-market seed offset.",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
        help="Use the same continuation-market seed rule as training validation or test.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the checkpoint filename for every selected policy.",
    )
    parser.add_argument(
        "--include-v26-controls",
        action="store_true",
        help=(
            "Include V2.6 Gaussian and Dirichlet best-return controls in every "
            "comparison plot, the branch CSV/JSON, and validation history."
        ),
    )
    return parser.parse_args()


def policy_specs_from_args(args: argparse.Namespace) -> list[PolicySpec]:
    policy_sets = {
        "default": DEFAULT_POLICIES,
        "v2.2": V22_POLICIES,
        "v2.3": V23_POLICIES,
        "v2.4": V24_POLICIES,
        "v2.5": V25_POLICIES,
        "v2.6": V26_POLICIES,
        "v3.2": V32_POLICIES,
    }
    specs = list(policy_sets[args.policy_set])
    if args.checkpoint:
        specs = [replace(spec, checkpoint=args.checkpoint) for spec in specs]
    if args.include_v26_controls:
        # V2.6 predates the independent best-feasible checkpoint. Keep these
        # reference policies on their preserved best-return snapshots even when
        # the selected V3.2 policies use checkpoint_best_feasible.pt.
        specs.extend(V26_CONTROL_POLICIES)
    if args.include_allocation_penalty:
        specs.append(
            PolicySpec(
                label="RCPO Allocation Penalty",
                run_dir=Path(args.allocation_run_dir),
            )
        )
    if args.include_allocation_drawdown_penalty:
        specs.append(
            PolicySpec(
                label="RCPO Allocation + Drawdown",
                run_dir=Path(args.allocation_drawdown_run_dir),
            )
        )
    return specs


def cumulative_return_path(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns) - 1.0


def annualized_return(returns: np.ndarray) -> float:
    log_growth = np.log1p(np.clip(returns, -0.999999, None))
    return float(np.exp(float(np.mean(log_growth)) * 252.0) - 1.0)


def annualized_volatility(returns: np.ndarray) -> float:
    return float(np.std(returns) * np.sqrt(252.0))


def rollout_policy(
    env: PortfolioEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray | float]:
    obs, _ = env.reset(options={"start_index": int(env.available_start_indices()[0])})
    returns: list[float] = []
    turnovers: list[float] = []
    constraint_costs: list[float] = []
    allocation_constraint_1_weights: list[float] = []
    allocation_constraint_2_weights: list[float] = []
    allocation_constraint_1_violations: list[float] = []
    allocation_constraint_2_violations: list[float] = []
    allocation_raw_costs: list[float] = []
    allocation_costs: list[float] = []
    effective_drawdown_budgets: list[float] = []
    weights: list[np.ndarray] = []
    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        del reward
        returns.append(float(info["net_return"]))
        turnovers.append(float(info["turnover"]))
        constraint_costs.append(float(info["constraint_cost"]))
        allocation_constraint_1_weights.append(float(info["allocation_constraint_1_weight"]))
        allocation_constraint_2_weights.append(float(info["allocation_constraint_2_weight"]))
        allocation_constraint_1_violations.append(
            float(info["allocation_constraint_1_violation_cost"])
        )
        allocation_constraint_2_violations.append(
            float(info["allocation_constraint_2_violation_cost"])
        )
        allocation_raw_costs.append(float(info["allocation_constraint_raw_cost"]))
        allocation_costs.append(float(info["allocation_constraint_cost"]))
        effective_drawdown_budgets.append(float(info["effective_drawdown_budget"]))
        weights.append(np.asarray(info["weights"], dtype=np.float32))
        if terminated or truncated:
            break
    returns_array = np.asarray(returns, dtype=np.float32)
    turnover_array = np.asarray(turnovers, dtype=np.float32)
    constraint_array = np.asarray(constraint_costs, dtype=np.float32)
    allocation_1_weight_array = np.asarray(allocation_constraint_1_weights, dtype=np.float32)
    allocation_2_weight_array = np.asarray(allocation_constraint_2_weights, dtype=np.float32)
    allocation_1_violation_array = np.asarray(
        allocation_constraint_1_violations,
        dtype=np.float32,
    )
    allocation_2_violation_array = np.asarray(
        allocation_constraint_2_violations,
        dtype=np.float32,
    )
    allocation_raw_cost_array = np.asarray(allocation_raw_costs, dtype=np.float32)
    allocation_cost_array = np.asarray(allocation_costs, dtype=np.float32)
    effective_budget_array = np.asarray(effective_drawdown_budgets, dtype=np.float32)
    weights_array = np.asarray(weights, dtype=np.float32)
    concentration_array = np.sum(np.square(weights_array), axis=1)
    max_single_asset_weight_array = np.max(weights_array, axis=1)
    return {
        "returns": returns_array,
        "cumulative_returns": cumulative_return_path(returns_array),
        "final_cumulative_return": float(cumulative_return_path(returns_array)[-1]),
        "annualized_return": annualized_return(returns_array),
        "annualized_volatility": annualized_volatility(returns_array),
        "max_drawdown": float(np.max(compute_drawdown(returns_array))),
        "average_turnover": float(np.mean(turnover_array)),
        "average_constraint_cost": float(np.mean(constraint_array)),
        "average_allocation_constraint_1_weight": float(np.mean(allocation_1_weight_array)),
        "average_allocation_constraint_2_weight": float(np.mean(allocation_2_weight_array)),
        "average_allocation_constraint_1_violation_cost": float(
            np.mean(allocation_1_violation_array)
        ),
        "average_allocation_constraint_2_violation_cost": float(
            np.mean(allocation_2_violation_array)
        ),
        "average_allocation_constraint_raw_cost": float(np.mean(allocation_raw_cost_array)),
        "average_allocation_constraint_cost": float(np.mean(allocation_cost_array)),
        "average_effective_drawdown_budget_squared": float(
            np.mean(np.square(effective_budget_array))
        ),
        "average_concentration": float(np.mean(concentration_array)),
        "maximum_concentration": float(np.max(concentration_array)),
        "average_max_single_asset_weight": float(np.mean(max_single_asset_weight_array)),
        "maximum_single_asset_weight": float(np.max(max_single_asset_weight_array)),
        "fraction_steps_single_asset_above_70pct": float(
            np.mean(max_single_asset_weight_array > 0.70)
        ),
        "average_weights": weights_array.mean(axis=0),
    }


def load_policy(spec: PolicySpec):
    config, metadata, model, environments = load_checkpoint_for_evaluation(
        spec.run_dir,
        checkpoint_name=spec.checkpoint,
    )
    device = next(model.parameters()).device

    def policy_fn(observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            output = model.get_policy_output(
                observation_tensor,
                deterministic=config.evaluation.deterministic,
            )
        return output.action.squeeze(0).cpu().numpy()

    return config, metadata, environments, policy_fn


def load_validation_history(run_dir: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    metrics_path = run_dir / "validation_metrics.jsonl"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return rows
    dedup: dict[int, dict[str, float]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "update" not in payload:
                continue
            relative_wealth = payload.get(
                "validation_mean_relative_wealth_vs_constrained_neutral"
            )
            if relative_wealth is None:
                model_return = payload.get("validation_mean_cumulative_return")
                baseline_return = payload.get(
                    "validation_equal_weight_mean_cumulative_return"
                )
                if model_return is None or baseline_return is None:
                    continue
                relative_wealth = (
                    (1.0 + float(model_return))
                    / max(1.0 + float(baseline_return), 1e-12)
                    - 1.0
                )
            if relative_wealth is None:
                continue
            payload["_display_relative_wealth"] = float(relative_wealth)
            dedup[int(payload["update"])] = payload
    for update in sorted(dedup):
        payload = dedup[update]
        rows.append(
            {
                "update": float(update),
                "validation_relative_wealth": float(payload["_display_relative_wealth"]),
            }
        )
    return rows


def write_branch_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "policy",
        "run_dir",
        "checkpoint",
        "branch_index",
        "final_cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
        "average_constraint_cost",
        "average_allocation_constraint_1_weight",
        "average_allocation_constraint_2_weight",
        "average_allocation_constraint_1_violation_cost",
        "average_allocation_constraint_2_violation_cost",
        "average_allocation_constraint_raw_cost",
        "average_allocation_constraint_cost",
        "average_concentration",
        "maximum_concentration",
        "average_max_single_asset_weight",
        "maximum_single_asset_weight",
        "fraction_steps_single_asset_above_70pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_cumulative_returns(
    path: Path,
    policy_returns: dict[str, list[np.ndarray]],
    baseline_returns: list[np.ndarray],
    *,
    show_branch_paths: bool = True,
    split_label: str = "Future",
) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(baseline_returns[0]))
    for label, paths in policy_returns.items():
        relative_paths = np.stack(
            [
                relative_wealth_path(policy_path, baseline_path)
                for policy_path, baseline_path in zip(
                    paths, baseline_returns, strict=True
                )
            ]
        )
        color = policy_color(label)
        if show_branch_paths:
            for branch_path in relative_paths:
                plt.plot(x, branch_path, color=color, alpha=0.035, linewidth=0.6)
        plt.plot(
            x,
            relative_paths.mean(axis=0),
            color=color,
            linewidth=2.2,
            label=label,
        )
    plt.axhline(
        0.0,
        color=POLICY_COLORS["constrained_neutral"],
        linestyle="--",
        linewidth=2.4,
        label="Constrained-neutral baseline",
    )
    title = (
        f"Section 9: {split_label} Relative Wealth Comparison"
        if show_branch_paths
        else f"Section 9: Mean {split_label} Relative Wealth Comparison"
    )
    plt.title(title)
    plt.xlabel(f"{split_label} Continuation Step")
    plt.ylabel("Relative Wealth vs Constrained-Neutral Baseline")
    apply_signed_percent_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bar_with_points(
    path: Path,
    title: str,
    ylabel: str,
    metric_by_policy: dict[str, list[float]],
) -> None:
    labels = list(metric_by_policy)
    means = [float(np.mean(metric_by_policy[label])) for label in labels]
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5.5))
    plt.bar(
        x,
        means,
        color=[policy_color(label) for label in labels],
        alpha=0.78,
    )
    rng = np.random.default_rng(123)
    for index, label in enumerate(labels):
        values = np.asarray(metric_by_policy[label], dtype=np.float64)
        jitter = rng.normal(0.0, 0.035, size=len(values))
        plt.scatter(
            np.full(len(values), index) + jitter,
            values,
            color=policy_color(label),
            alpha=0.6,
            s=20,
            edgecolors="#202020",
            linewidths=0.25,
        )
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_validation_history(
    path: Path,
    histories: dict[str, list[dict[str, float]]],
    *,
    max_update: float | None = None,
) -> None:
    plt.figure(figsize=(10, 5.5))
    for label, rows in histories.items():
        if max_update is not None:
            rows = [row for row in rows if row["update"] <= max_update]
        if not rows:
            continue
        updates = [row["update"] for row in rows]
        scores = [row["validation_relative_wealth"] for row in rows]
        plt.plot(
            updates,
            scores,
            color=policy_color(label),
            linewidth=1.8,
            label=label,
        )
    plt.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.5)
    plt.title("Section 9: Validation Relative Wealth History")
    plt.xlabel("Training Update")
    plt.ylabel("Relative Wealth vs Constrained-Neutral Baseline")
    apply_signed_percent_axis(plt.gca())
    if max_update is not None:
        plt.xlim(left=0.0, right=max_update)
    if any(rows for rows in histories.values()):
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_return_risk_feasibility_pareto(
    path: Path,
    summaries: dict[str, dict[str, object]],
) -> None:
    labels = list(summaries)
    returns = np.asarray(
        [
            float(summaries[label].get("mean_relative_wealth_vs_constrained_neutral", 0.0))
            for label in labels
        ]
    )
    drawdowns = np.asarray(
        [float(summaries[label]["mean_max_drawdown"]) for label in labels]
    )
    feasible = np.asarray(
        [bool(summaries[label]["active_constraint_feasible"]) for label in labels]
    )
    concentrations = np.asarray(
        [float(summaries[label]["mean_concentration"]) for label in labels]
    )
    # Show the return-risk frontier only among policies satisfying their active constraints.
    frontier_indices: list[int] = []
    for index in np.flatnonzero(feasible):
        dominated = any(
            other != index
            and feasible[other]
            and returns[other] >= returns[index]
            and drawdowns[other] <= drawdowns[index]
            and (
                returns[other] > returns[index]
                or drawdowns[other] < drawdowns[index]
            )
            for other in range(len(labels))
        )
        if not dominated:
            frontier_indices.append(int(index))

    plt.figure(figsize=(10.5, 6.2))
    if frontier_indices:
        frontier_indices.sort(key=lambda index: drawdowns[index])
        plt.plot(
            drawdowns[frontier_indices],
            returns[frontier_indices],
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

    for index, label in enumerate(labels):
        color = policy_color(label)
        size = 100.0 + 450.0 * concentrations[index]
        plt.scatter(
            drawdowns[index],
            returns[index],
            s=size,
            facecolors=color if feasible[index] else "none",
            edgecolors=color,
            linewidths=2.0,
            zorder=3,
        )
        plt.annotate(
            label,
            (drawdowns[index], returns[index]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#777777",
            markeredgecolor="#333333", markersize=9, label="Active constraint feasible",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none",
            markeredgecolor="#333333", markeredgewidth=1.8, markersize=9,
            label="Active constraint violation",
        ),
        Line2D(
            [0], [0], color="#777777", linestyle="--", linewidth=1.2,
            label="Feasible return-risk frontier",
        ),
    ]
    plt.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.5)
    plt.title("Section 9: Return-Risk-Feasibility Pareto Comparison")
    plt.xlabel("Mean Maximum Drawdown (lower is better)")
    plt.ylabel("Relative Wealth vs Constrained-Neutral Baseline")
    apply_signed_percent_axis(plt.gca())
    plt.figtext(
        0.5,
        0.01,
        "Marker area represents mean concentration; feasibility refers to each method's active constraints.",
        ha="center",
        fontsize=9,
    )
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    policy_specs = policy_specs_from_args(args)
    for spec in policy_specs:
        config, metadata, environments, policy_fn = load_policy(spec)
        loaded.append((spec, config, metadata, environments, policy_fn))

    anchor_spec, anchor_config, anchor_metadata, anchor_envs, _ = loaded[0]
    split_name = args.evaluation_split
    split_label = "Validation" if split_name == "validation" else "Test"
    steps = (
        anchor_config.market.validation_steps
        if split_name == "validation"
        else anchor_config.market.test_steps
    )
    configured_market_count = (
        anchor_config.evaluation.validation_branch_count
        if split_name == "validation"
        else anchor_config.evaluation.test_branch_count
    )
    market_count = (
        int(args.future_market_count)
        if args.future_market_count is not None
        else int(configured_market_count)
    )
    seed_offset = (
        int(args.future_seed_offset)
        if args.future_seed_offset is not None
        else (10_000 if split_name == "validation" else 20_000)
    )
    future_markets = generate_continuation_splits(
        anchor_config.market,
        anchor_envs["train"].market,
        steps,
        int(anchor_metadata["seed"]) + seed_offset,
        count=market_count,
    )

    policy_return_paths: dict[str, list[np.ndarray]] = {}
    max_drawdowns: dict[str, list[float]] = {}
    turnovers: dict[str, list[float]] = {}
    concentrations: dict[str, list[float]] = {}
    csv_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    baseline_returns: list[np.ndarray] = []
    baseline_metrics: list[dict[str, np.ndarray | float]] = []

    for branch_index, market in enumerate(future_markets):
        baseline_env = PortfolioEnv(
            anchor_config.environment,
            market,
            anchor_config.market,
            seed=int(anchor_metadata["seed"]) + seed_offset + branch_index,
        )
        baseline_result = rollout_policy(
            baseline_env,
            lambda _obs, env=baseline_env: env.constrained_neutral_action(),
        )
        baseline_returns.append(baseline_result["returns"])  # type: ignore[arg-type]
        baseline_metrics.append(baseline_result)

    max_drawdowns["Constrained-neutral baseline"] = [
        float(metric["max_drawdown"]) for metric in baseline_metrics
    ]
    turnovers["Constrained-neutral baseline"] = [
        float(metric["average_turnover"]) for metric in baseline_metrics
    ]
    concentrations["Constrained-neutral baseline"] = [
        float(metric["average_concentration"]) for metric in baseline_metrics
    ]
    for branch_index, metric in enumerate(baseline_metrics):
        csv_rows.append(
            {
                "policy": "Constrained-neutral baseline",
                "run_dir": str(anchor_spec.run_dir),
                "checkpoint": "neutral_action",
                "branch_index": branch_index,
                "final_cumulative_return": metric["final_cumulative_return"],
                "annualized_return": metric["annualized_return"],
                "annualized_volatility": metric["annualized_volatility"],
                "max_drawdown": metric["max_drawdown"],
                "average_turnover": metric["average_turnover"],
                "average_constraint_cost": metric["average_constraint_cost"],
                "average_allocation_constraint_1_weight": metric[
                    "average_allocation_constraint_1_weight"
                ],
                "average_allocation_constraint_2_weight": metric[
                    "average_allocation_constraint_2_weight"
                ],
                "average_allocation_constraint_1_violation_cost": metric[
                    "average_allocation_constraint_1_violation_cost"
                ],
                "average_allocation_constraint_2_violation_cost": metric[
                    "average_allocation_constraint_2_violation_cost"
                ],
                "average_allocation_constraint_raw_cost": metric[
                    "average_allocation_constraint_raw_cost"
                ],
                "average_allocation_constraint_cost": metric[
                    "average_allocation_constraint_cost"
                ],
                "average_concentration": metric["average_concentration"],
                "maximum_concentration": metric["maximum_concentration"],
                "average_max_single_asset_weight": metric[
                    "average_max_single_asset_weight"
                ],
                "maximum_single_asset_weight": metric["maximum_single_asset_weight"],
                "fraction_steps_single_asset_above_70pct": metric[
                    "fraction_steps_single_asset_above_70pct"
                ],
            }
        )

    validation_histories: dict[str, list[dict[str, float]]] = {}
    for spec, config, metadata, _environments, policy_fn in loaded:
        returns_for_policy: list[np.ndarray] = []
        max_drawdowns[spec.label] = []
        turnovers[spec.label] = []
        concentrations[spec.label] = []
        branch_metrics: list[dict[str, np.ndarray | float]] = []
        validation_histories[spec.label] = load_validation_history(spec.run_dir)
        for branch_index, market in enumerate(future_markets):
            env = PortfolioEnv(
                config.environment,
                market,
                config.market,
                seed=int(metadata["seed"]) + seed_offset + branch_index,
            )
            result = rollout_policy(env, policy_fn)
            returns_for_policy.append(result["returns"])  # type: ignore[arg-type]
            max_drawdowns[spec.label].append(float(result["max_drawdown"]))
            turnovers[spec.label].append(float(result["average_turnover"]))
            concentrations[spec.label].append(float(result["average_concentration"]))
            branch_metrics.append(result)
            csv_rows.append(
                {
                    "policy": spec.label,
                    "run_dir": str(spec.run_dir),
                    "checkpoint": spec.checkpoint,
                    "branch_index": branch_index,
                    "final_cumulative_return": result["final_cumulative_return"],
                    "annualized_return": result["annualized_return"],
                    "annualized_volatility": result["annualized_volatility"],
                    "max_drawdown": result["max_drawdown"],
                    "average_turnover": result["average_turnover"],
                    "average_constraint_cost": result["average_constraint_cost"],
                    "average_allocation_constraint_1_weight": result[
                        "average_allocation_constraint_1_weight"
                    ],
                    "average_allocation_constraint_2_weight": result[
                        "average_allocation_constraint_2_weight"
                    ],
                    "average_allocation_constraint_1_violation_cost": result[
                        "average_allocation_constraint_1_violation_cost"
                    ],
                    "average_allocation_constraint_2_violation_cost": result[
                        "average_allocation_constraint_2_violation_cost"
                    ],
                    "average_allocation_constraint_raw_cost": result[
                        "average_allocation_constraint_raw_cost"
                    ],
                    "average_allocation_constraint_cost": result[
                        "average_allocation_constraint_cost"
                    ],
                    "average_concentration": result["average_concentration"],
                    "maximum_concentration": result["maximum_concentration"],
                    "average_max_single_asset_weight": result[
                        "average_max_single_asset_weight"
                    ],
                    "maximum_single_asset_weight": result[
                        "maximum_single_asset_weight"
                    ],
                    "fraction_steps_single_asset_above_70pct": result[
                        "fraction_steps_single_asset_above_70pct"
                    ],
                }
            )
        policy_return_paths[spec.label] = returns_for_policy
        baseline_final = np.asarray(
            [metric["final_cumulative_return"] for metric in baseline_metrics],
            dtype=np.float64,
        )
        policy_final = np.asarray(
            [metric["final_cumulative_return"] for metric in branch_metrics],
            dtype=np.float64,
        )
        is_rcpo = str(metadata.get("algo", "")) == "rcpo"
        allocation_feasible_by_branch = np.asarray(
            [
                float(metric["average_allocation_constraint_raw_cost"]) <= 1e-8
                for metric in branch_metrics
            ],
            dtype=bool,
        )
        if is_rcpo:
            if config.rcpo.alpha is not None:
                constraint_alphas = np.full(
                    len(branch_metrics),
                    float(config.rcpo.alpha),
                    dtype=np.float64,
                )
            else:
                ratio = float(config.rcpo.alpha_budget_ratio)
                scale = float(config.environment.drawdown_cost_scale)
                constraint_alphas = np.asarray(
                    [
                        ratio**2
                        * float(metric["average_effective_drawdown_budget_squared"])
                        / scale
                        for metric in branch_metrics
                    ],
                    dtype=np.float64,
                )
            constraint_costs_for_policy = np.asarray(
                [float(metric["average_constraint_cost"]) for metric in branch_metrics],
                dtype=np.float64,
            )
            feasible_by_branch = allocation_feasible_by_branch & (
                constraint_costs_for_policy <= constraint_alphas + 1e-12
            )
            mean_constraint_alpha: float | None = float(np.mean(constraint_alphas))
            active_constraint_feasible = bool(
                np.mean(constraint_costs_for_policy)
                <= mean_constraint_alpha + 1e-12
                and np.all(allocation_feasible_by_branch)
            )
        else:
            feasible_by_branch = allocation_feasible_by_branch
            mean_constraint_alpha = None
            active_constraint_feasible = bool(np.all(allocation_feasible_by_branch))

        summaries[spec.label] = {
            "run_dir": str(spec.run_dir),
            "checkpoint": spec.checkpoint,
            "policy_architecture": config.network.policy_architecture,
            "branch_credit_mode": config.network.branch_credit_mode,
            "mean_final_cumulative_return": float(np.mean(policy_final)),
            "mean_excess_cumulative_return": float(np.mean(policy_final - baseline_final)),
            "mean_relative_wealth_vs_constrained_neutral": float(
                np.mean(
                    (1.0 + policy_final)
                    / np.maximum(1.0 + baseline_final, 1e-12)
                    - 1.0
                )
            ),
            "win_rate_vs_constrained_neutral": float(np.mean(policy_final > baseline_final)),
            "mean_annualized_return": float(
                np.mean([metric["annualized_return"] for metric in branch_metrics])
            ),
            "mean_annualized_volatility": float(
                np.mean([metric["annualized_volatility"] for metric in branch_metrics])
            ),
            "mean_max_drawdown": float(np.mean(max_drawdowns[spec.label])),
            "mean_average_turnover": float(np.mean(turnovers[spec.label])),
            "mean_concentration": float(np.mean(concentrations[spec.label])),
            "mean_maximum_concentration": float(
                np.mean([metric["maximum_concentration"] for metric in branch_metrics])
            ),
            "mean_max_single_asset_weight": float(
                np.mean([metric["average_max_single_asset_weight"] for metric in branch_metrics])
            ),
            "worst_case_single_asset_weight": float(
                np.max([metric["maximum_single_asset_weight"] for metric in branch_metrics])
            ),
            "mean_fraction_steps_single_asset_above_70pct": float(
                np.mean([
                    metric["fraction_steps_single_asset_above_70pct"]
                    for metric in branch_metrics
                ])
            ),
            "mean_constraint_cost": float(
                np.mean([metric["average_constraint_cost"] for metric in branch_metrics])
            ),
            "mean_constraint_alpha": mean_constraint_alpha,
            "active_constraint_feasible": active_constraint_feasible,
            "constraint_feasible_branch_rate": float(np.mean(feasible_by_branch)),
            "mean_allocation_constraint_raw_cost": float(
                np.mean(
                    [
                        metric["average_allocation_constraint_raw_cost"]
                        for metric in branch_metrics
                    ]
                )
            ),
            "mean_allocation_constraint_cost": float(
                np.mean(
                    [
                        metric["average_allocation_constraint_cost"]
                        for metric in branch_metrics
                    ]
                )
            ),
            "mean_allocation_constraint_1_violation_cost": float(
                np.mean(
                    [
                        metric["average_allocation_constraint_1_violation_cost"]
                        for metric in branch_metrics
                    ]
                )
            ),
            "mean_allocation_constraint_2_violation_cost": float(
                np.mean(
                    [
                        metric["average_allocation_constraint_2_violation_cost"]
                        for metric in branch_metrics
                    ]
                )
            ),
        }

    v32_history_max_update = max(
        (
            row["update"]
            for rows in validation_histories.values()
            for row in rows
        ),
        default=None,
    )
    baseline_final = np.asarray(
        [metric["final_cumulative_return"] for metric in baseline_metrics],
        dtype=np.float64,
    )
    summaries["Constrained-neutral baseline"] = {
        "run_dir": str(anchor_spec.run_dir),
        "checkpoint": "neutral_action",
        "mean_final_cumulative_return": float(np.mean(baseline_final)),
        "mean_excess_cumulative_return": 0.0,
        "mean_relative_wealth_vs_constrained_neutral": 0.0,
        "mean_annualized_return": float(
            np.mean([metric["annualized_return"] for metric in baseline_metrics])
        ),
        "mean_annualized_volatility": float(
            np.mean([metric["annualized_volatility"] for metric in baseline_metrics])
        ),
        "mean_max_drawdown": float(np.mean(max_drawdowns["Constrained-neutral baseline"])),
        "mean_average_turnover": float(np.mean(turnovers["Constrained-neutral baseline"])),
        "mean_concentration": float(np.mean(concentrations["Constrained-neutral baseline"])),
        "mean_maximum_concentration": float(
            np.mean([metric["maximum_concentration"] for metric in baseline_metrics])
        ),
        "mean_max_single_asset_weight": float(
            np.mean([metric["average_max_single_asset_weight"] for metric in baseline_metrics])
        ),
        "worst_case_single_asset_weight": float(
            np.max([metric["maximum_single_asset_weight"] for metric in baseline_metrics])
        ),
        "mean_fraction_steps_single_asset_above_70pct": float(
            np.mean([
                metric["fraction_steps_single_asset_above_70pct"]
                for metric in baseline_metrics
            ])
        ),
        "mean_constraint_cost": float(
            np.mean([metric["average_constraint_cost"] for metric in baseline_metrics])
        ),
        "mean_constraint_alpha": None,
        "active_constraint_feasible": True,
        "constraint_feasible_branch_rate": 1.0,
        "mean_allocation_constraint_raw_cost": float(
            np.mean([metric["average_allocation_constraint_raw_cost"] for metric in baseline_metrics])
        ),
        "mean_allocation_constraint_cost": float(
            np.mean([metric["average_allocation_constraint_cost"] for metric in baseline_metrics])
        ),
    }

    plot_cumulative_returns(
        output_dir / "section9_cumulative_return_comparison.png",
        policy_return_paths,
        baseline_returns,
        split_label=split_label,
    )
    plot_cumulative_returns(
        output_dir / "section9_cumulative_return_means_only.png",
        policy_return_paths,
        baseline_returns,
        show_branch_paths=False,
        split_label=split_label,
    )
    plot_bar_with_points(
        output_dir / "section9_max_drawdown_comparison.png",
        "Section 9: Maximum Drawdown Comparison",
        "Maximum Drawdown (lower is better)",
        max_drawdowns,
    )
    plot_bar_with_points(
        output_dir / "section9_turnover_comparison.png",
        "Section 9: Turnover Comparison",
        "Average Turnover",
        turnovers,
    )
    plot_bar_with_points(
        output_dir / "section9_concentration_comparison.png",
        "Section 9: Portfolio Concentration Comparison",
        "Mean sum of squared portfolio weights (lower is more diversified)",
        concentrations,
    )
    plot_validation_history(
        output_dir / "section9_validation_score_history.png",
        validation_histories,
        max_update=v32_history_max_update,
    )
    plot_return_risk_feasibility_pareto(
        output_dir / "section9_return_risk_feasibility_pareto.png",
        summaries,
    )
    write_branch_metrics(output_dir / "section9_branch_metrics.csv", csv_rows)
    with (output_dir / "section9_comparison_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "evaluation_split": split_name,
                "continuation_market_count": market_count,
                "continuation_seed_offset": seed_offset,
                "continuation_steps": int(steps),
                "anchor_run_dir": str(anchor_spec.run_dir),
                "anchor_seed": int(anchor_metadata["seed"]),
                "baseline": "constrained_neutral",
                "policies": summaries,
            },
            handle,
            indent=2,
        )
    print(f"Wrote Section 9 comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
