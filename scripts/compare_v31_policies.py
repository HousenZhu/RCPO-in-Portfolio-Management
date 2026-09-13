from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compare_section9_simplex import (  # noqa: E402
    PolicySpec,
    load_policy,
    load_validation_history,
    rollout_policy,
)
from rcpo_portfolio.env import PortfolioEnv  # noqa: E402
from rcpo_portfolio.evaluation import (  # noqa: E402
    apply_signed_percent_axis,
    relative_wealth_path,
)
from rcpo_portfolio.market import generate_continuation_splits  # noqa: E402


METHOD_RUNS: dict[str, dict[int, Path]] = {
    "Q80 lambda-up 0.00025": {
        3: PROJECT_ROOT
        / "runs"
        / "simplex_v3.1_final_q80_lr025_rcpo_none_20260831_200122"
        / "seed_3",
        4: PROJECT_ROOT
        / "runs"
        / "simplex_v3.1_final_q80_lr025_rcpo_none_20260831_200126"
        / "seed_4",
    },
    "Q80 lambda-up 0.00040": {
        3: PROJECT_ROOT
        / "runs"
        / "simplex_v3.1_final_q80_lr040_rcpo_none_20260831_200130"
        / "seed_3",
        4: PROJECT_ROOT
        / "runs"
        / "simplex_v3.1_final_q80_lr040_rcpo_none_20260831_200134"
        / "seed_4",
    },
    "Mean-step control": {
        3: PROJECT_ROOT
        / "runs"
        / "simplex_v3.1_control_mean_step_rcpo_none_20260831_200138"
        / "seed_3",
        4: PROJECT_ROOT
        / "runs"
        / "simplex_v3.1_control_mean_step_rcpo_none_20260831_200142"
        / "seed_4",
    },
}

COLORS = {
    "Q80 lambda-up 0.00025": "#2474A6",
    "Q80 lambda-up 0.00040": "#E67E22",
    "Mean-step control": "#238B57",
    "Constrained-neutral baseline": "#151515",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V3.1 RCPO policies across paired seed-specific future markets."
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "simplex_v3.1_policy_comparison"),
    )
    parser.add_argument("--future-market-count", type=int, default=20)
    parser.add_argument("--future-seed-offset", type=int, default=20_000)
    return parser.parse_args()


def _metric_row(
    policy: str,
    run_dir: Path,
    checkpoint: str,
    training_seed: int,
    branch_index: int,
    result: dict[str, np.ndarray | float],
) -> dict[str, object]:
    return {
        "policy": policy,
        "training_seed": training_seed,
        "branch_index": branch_index,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "final_cumulative_return": result["final_cumulative_return"],
        "annualized_return": result["annualized_return"],
        "annualized_volatility": result["annualized_volatility"],
        "max_drawdown": result["max_drawdown"],
        "average_turnover": result["average_turnover"],
        "average_constraint_cost": result["average_constraint_cost"],
        "average_effective_drawdown_budget_squared": result[
            "average_effective_drawdown_budget_squared"
        ],
        "average_allocation_constraint_raw_cost": result[
            "average_allocation_constraint_raw_cost"
        ],
        "average_concentration": result["average_concentration"],
        "maximum_single_asset_weight": result["maximum_single_asset_weight"],
    }


def _plot_relative_wealth(
    output_path: Path,
    policy_paths: dict[str, list[np.ndarray]],
    baseline_paths: list[np.ndarray],
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(baseline_paths[0]))
    for label, paths in policy_paths.items():
        relative_paths = np.stack(
            [
                relative_wealth_path(path, baseline)
                for path, baseline in zip(paths, baseline_paths, strict=True)
            ]
        )
        for path in relative_paths:
            axis.plot(x, path, color=COLORS[label], linewidth=0.6, alpha=0.035)
        axis.plot(
            x,
            relative_paths.mean(axis=0),
            color=COLORS[label],
            linewidth=2.4,
            label=label,
        )
    axis.axhline(
        0.0,
        color=COLORS["Constrained-neutral baseline"],
        linestyle="--",
        linewidth=2.2,
        label="Constrained-neutral baseline",
    )
    axis.set_title("V3.1 Test Relative Wealth Comparison")
    axis.set_xlabel("Future Market Step")
    axis.set_ylabel("Relative Wealth vs Constrained-Neutral Baseline")
    apply_signed_percent_axis(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_bar_points(
    output_path: Path,
    title: str,
    ylabel: str,
    values: dict[str, list[float]],
    percent_axis: bool = False,
) -> None:
    labels = list(values)
    x = np.arange(len(labels))
    means = [float(np.mean(values[label])) for label in labels]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.bar(x, means, color=[COLORS[label] for label in labels], alpha=0.80)
    rng = np.random.default_rng(31)
    for index, label in enumerate(labels):
        points = np.asarray(values[label], dtype=np.float64)
        axis.scatter(
            np.full(len(points), index) + rng.normal(0.0, 0.035, len(points)),
            points,
            color=COLORS[label],
            edgecolors="#222222",
            linewidths=0.25,
            s=21,
            alpha=0.62,
        )
    axis.set_xticks(x, labels, rotation=16, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if percent_axis:
        apply_signed_percent_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _aggregate_histories(
    histories: dict[str, list[list[dict[str, float]]]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, method_histories in histories.items():
        by_update: dict[int, list[float]] = {}
        for history in method_histories:
            for row in history:
                update = int(row["update"])
                by_update.setdefault(update, []).append(
                    float(row["validation_relative_wealth"])
                )
        updates = np.asarray(sorted(by_update), dtype=np.int64)
        means = np.asarray(
            [np.mean(by_update[int(update)]) for update in updates], dtype=np.float64
        )
        stds = np.asarray(
            [np.std(by_update[int(update)]) for update in updates], dtype=np.float64
        )
        output[label] = updates, means, stds
    return output


def _plot_validation_history(
    output_path: Path,
    histories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for label, (updates, means, stds) in histories.items():
        axis.plot(updates + 1, means, color=COLORS[label], linewidth=2.0, label=label)
        axis.fill_between(
            updates + 1,
            means - stds,
            means + stds,
            color=COLORS[label],
            alpha=0.10,
        )
    axis.axhline(0.0, color="#777777", linewidth=0.9)
    axis.set_title("V3.1 Validation Relative Wealth History")
    axis.set_xlabel("Training Update")
    axis.set_ylabel("Relative Wealth vs Constrained-Neutral Baseline")
    apply_signed_percent_axis(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _summarize_method(
    results: list[dict[str, np.ndarray | float]],
    paths: list[np.ndarray],
    baseline_results: list[dict[str, np.ndarray | float]],
    baseline_paths: list[np.ndarray],
    alpha_budget_ratio: float,
    drawdown_cost_scale: float,
) -> dict[str, float | int | bool]:
    relative_final = np.asarray(
        [
            relative_wealth_path(path, baseline)[-1]
            for path, baseline in zip(paths, baseline_paths, strict=True)
        ],
        dtype=np.float64,
    )
    costs = np.asarray(
        [float(result["average_constraint_cost"]) for result in results]
    )
    alphas = np.asarray(
        [
            alpha_budget_ratio**2
            * float(result["average_effective_drawdown_budget_squared"])
            / drawdown_cost_scale
            for result in results
        ]
    )
    allocation_feasible = np.asarray(
        [
            float(result["average_allocation_constraint_raw_cost"]) <= 1e-8
            for result in results
        ]
    )
    feasible = allocation_feasible & (costs <= alphas + 1e-12)
    baseline_final = np.asarray(
        [float(result["final_cumulative_return"]) for result in baseline_results]
    )
    policy_final = np.asarray(
        [float(result["final_cumulative_return"]) for result in results]
    )
    return {
        "evaluation_paths": len(results),
        "mean_relative_wealth_vs_constrained_neutral": float(np.mean(relative_final)),
        "relative_wealth_std": float(np.std(relative_final)),
        "win_rate_vs_constrained_neutral": float(np.mean(relative_final > 0.0)),
        "mean_excess_cumulative_return": float(np.mean(policy_final - baseline_final)),
        "mean_annualized_return": float(
            np.mean([float(result["annualized_return"]) for result in results])
        ),
        "mean_annualized_volatility": float(
            np.mean([float(result["annualized_volatility"]) for result in results])
        ),
        "mean_max_drawdown": float(
            np.mean([float(result["max_drawdown"]) for result in results])
        ),
        "mean_average_turnover": float(
            np.mean([float(result["average_turnover"]) for result in results])
        ),
        "mean_concentration": float(
            np.mean([float(result["average_concentration"]) for result in results])
        ),
        "mean_constraint_cost": float(np.mean(costs)),
        "mean_alpha": float(np.mean(alphas)),
        "constraint_feasible_branch_rate": float(np.mean(feasible)),
        "active_constraint_feasible": bool(np.all(feasible)),
    }


def main() -> None:
    args = parse_args()
    if args.future_market_count < 1:
        raise ValueError("--future-market-count must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    histories: dict[str, list[list[dict[str, float]]]] = {
        label: [] for label in METHOD_RUNS
    }
    for label, seed_runs in METHOD_RUNS.items():
        for seed, run_dir in seed_runs.items():
            spec = PolicySpec(label=label, run_dir=run_dir)
            config, metadata, environments, policy_fn = load_policy(spec)
            if int(metadata["seed"]) != seed:
                raise ValueError(f"Unexpected checkpoint seed for {run_dir}.")
            loaded[(label, seed)] = (spec, config, metadata, environments, policy_fn)
            histories[label].append(load_validation_history(run_dir))

    baseline_paths: list[np.ndarray] = []
    baseline_results: list[dict[str, np.ndarray | float]] = []
    policy_paths: dict[str, list[np.ndarray]] = {label: [] for label in METHOD_RUNS}
    policy_results: dict[str, list[dict[str, np.ndarray | float]]] = {
        label: [] for label in METHOD_RUNS
    }
    csv_rows: list[dict[str, object]] = []

    seeds = sorted(next(iter(METHOD_RUNS.values())))
    anchor_label = next(iter(METHOD_RUNS))
    for seed in seeds:
        _, anchor_config, anchor_metadata, anchor_envs, _ = loaded[(anchor_label, seed)]
        future_markets = generate_continuation_splits(
            anchor_config.market,
            anchor_envs["train"].market,
            anchor_config.market.test_steps,
            seed + int(args.future_seed_offset),
            count=int(args.future_market_count),
        )
        seed_baselines: list[dict[str, np.ndarray | float]] = []
        for branch_index, market in enumerate(future_markets):
            env = PortfolioEnv(
                anchor_config.environment,
                market,
                anchor_config.market,
                seed=seed + int(args.future_seed_offset) + branch_index,
            )
            result = rollout_policy(
                env,
                lambda _obs, current_env=env: current_env.constrained_neutral_action(),
            )
            seed_baselines.append(result)
            baseline_results.append(result)
            baseline_paths.append(result["returns"])  # type: ignore[arg-type]
            csv_rows.append(
                _metric_row(
                    "Constrained-neutral baseline",
                    Path("seed-specific-anchor"),
                    "neutral_action",
                    seed,
                    branch_index,
                    result,
                )
            )

        for label in METHOD_RUNS:
            spec, config, metadata, _, policy_fn = loaded[(label, seed)]
            del metadata
            for branch_index, market in enumerate(future_markets):
                env = PortfolioEnv(
                    config.environment,
                    market,
                    config.market,
                    seed=seed + int(args.future_seed_offset) + branch_index,
                )
                result = rollout_policy(env, policy_fn)
                policy_results[label].append(result)
                policy_paths[label].append(result["returns"])  # type: ignore[arg-type]
                csv_rows.append(
                    _metric_row(
                        label,
                        spec.run_dir,
                        spec.checkpoint,
                        seed,
                        branch_index,
                        result,
                    )
                )

    summaries: dict[str, dict[str, object]] = {}
    for label in METHOD_RUNS:
        _, config, _, _, _ = loaded[(label, seeds[0])]
        summaries[label] = {
            "checkpoint": "checkpoint_best_return.pt",
            "training_seeds": seeds,
            **_summarize_method(
                policy_results[label],
                policy_paths[label],
                baseline_results,
                baseline_paths,
                float(config.rcpo.alpha_budget_ratio),
                float(config.environment.drawdown_cost_scale),
            ),
        }

    _, baseline_config, _, _, _ = loaded[(anchor_label, seeds[0])]
    baseline_summary = _summarize_method(
        baseline_results,
        baseline_paths,
        baseline_results,
        baseline_paths,
        float(baseline_config.rcpo.alpha_budget_ratio),
        float(baseline_config.environment.drawdown_cost_scale),
    )
    summaries["Constrained-neutral baseline"] = {
        "checkpoint": "neutral_action",
        "training_seeds": seeds,
        **baseline_summary,
    }

    _plot_relative_wealth(
        output_dir / "v31_cumulative_return_comparison.png",
        policy_paths,
        baseline_paths,
    )
    drawdowns = {
        label: [float(result["max_drawdown"]) for result in policy_results[label]]
        for label in METHOD_RUNS
    }
    drawdowns["Constrained-neutral baseline"] = [
        float(result["max_drawdown"]) for result in baseline_results
    ]
    _plot_bar_points(
        output_dir / "v31_max_drawdown_comparison.png",
        "V3.1 Test Maximum Drawdown Comparison",
        "Maximum Drawdown (lower is better)",
        drawdowns,
        percent_axis=True,
    )
    turnovers = {
        label: [float(result["average_turnover"]) for result in policy_results[label]]
        for label in METHOD_RUNS
    }
    turnovers["Constrained-neutral baseline"] = [
        float(result["average_turnover"]) for result in baseline_results
    ]
    _plot_bar_points(
        output_dir / "v31_turnover_comparison.png",
        "V3.1 Test Turnover Comparison",
        "Average Turnover",
        turnovers,
    )
    feasibility = {
        label: [float(summaries[label]["constraint_feasible_branch_rate"])]
        for label in METHOD_RUNS
    }
    feasibility["Constrained-neutral baseline"] = [
        float(summaries["Constrained-neutral baseline"]["constraint_feasible_branch_rate"])
    ]
    _plot_bar_points(
        output_dir / "v31_constraint_feasibility_comparison.png",
        "V3.1 Test Constraint Feasibility Comparison",
        "Feasible Future-Market Rate",
        feasibility,
        percent_axis=True,
    )
    _plot_validation_history(
        output_dir / "v31_validation_score_history.png",
        _aggregate_histories(histories),
    )

    fieldnames = list(csv_rows[0])
    with (output_dir / "v31_branch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with (output_dir / "v31_comparison_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "evaluation_design": (
                    "Each training seed is evaluated on 20 continuations from its own "
                    "anchor train endpoint; method summaries pool both paired seeds."
                ),
                "checkpoint": "checkpoint_best_return.pt",
                "training_seeds": seeds,
                "future_markets_per_seed": int(args.future_market_count),
                "future_seed_offset": int(args.future_seed_offset),
                "total_future_paths_per_method": len(baseline_results),
                "baseline": "constrained_neutral",
                "policies": summaries,
            },
            handle,
            indent=2,
        )
    print(f"Wrote V3.1 comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
