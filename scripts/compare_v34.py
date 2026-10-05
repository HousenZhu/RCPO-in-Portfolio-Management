from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_section9_simplex import PolicySpec, load_policy, rollout_policy
from rcpo_portfolio.config import load_config
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.evaluation import relative_wealth_path
from rcpo_portfolio.market import generate_continuation_split, generate_market_split

CONFIG_DIR = ROOT / "configs" / "v3.4_experiment2_8assets"
CONTROL_PATTERNS = {
    ("base.yaml", 0): "simplex_v3.3_rcpo_gaussian_m2_s5040_e3_rcpo_none_*/seed_0",
    ("standalone_m8_e3.yaml", 0): "simplex_v3.3_rcpo_gaussian_m8_s5040_e3_rcpo_none_*/seed_0",
}
CONDITIONS = {
    "base.yaml": "Standalone 2x5040 e3",
    "standalone_m2_e4.yaml": "Standalone 2x5040 e4",
    "standalone_m8_e3.yaml": "Standalone 8x5040 e3",
    "standalone_m8_e4.yaml": "Standalone 8x5040 e4",
    "cf_reward_m2_e3.yaml": "Counterfactual 2x5040 e3",
    "cf_reward_m2_e4.yaml": "Counterfactual 2x5040 e4",
    "window_20.yaml": "Window 20",
    "window_30.yaml": "Window 30",
    "window_40.yaml": "Window 40",
    "window_60.yaml": "Window 60",
    "margin_085.yaml": "Margin 0.85",
    "margin_080.yaml": "Margin 0.80",
}
GROUPS = {
    "factorial": [
        "base.yaml", "standalone_m2_e4.yaml", "standalone_m8_e3.yaml",
        "standalone_m8_e4.yaml",
    ],
    "counterfactual": [
        "base.yaml", "standalone_m2_e4.yaml", "cf_reward_m2_e3.yaml",
        "cf_reward_m2_e4.yaml",
    ],
    "window": ["window_20.yaml", "window_30.yaml", "window_40.yaml", "window_60.yaml"],
    "margin": ["standalone_m2_e4.yaml", "margin_085.yaml", "margin_080.yaml"],
}


@dataclass(frozen=True)
class RunSpec:
    task_id: str
    config_name: str
    seed: int
    run_dir: Path


def manifest_specs(runs_root: Path, *, controls_only: bool = False) -> list[RunSpec]:
    specs: list[RunSpec] = []
    with (CONFIG_DIR / "tasks.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if controls_only:
            continue
        matches = list((runs_root / row["task_id"]).glob(f"*/seed_{row['seed']}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one run for {row['task_id']}, found {len(matches)} under {runs_root}"
            )
        specs.append(RunSpec(row["task_id"], row["config"], int(row["seed"]), matches[0]))
    for (config_name, seed), pattern in CONTROL_PATTERNS.items():
        matches = list((ROOT / "runs").glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one V3.3 control matching {pattern}, found {len(matches)}")
        specs.append(RunSpec(f"v33_control_{config_name[:-5]}", config_name, seed, matches[0]))
    return specs


def common_drawdown_metrics(
    agent_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    margin: float = 0.90,
    floor: float = 0.05,
    scale: float = 0.10,
    alpha_budget_ratio: float = 0.05,
) -> dict[str, float]:
    agent_dd = drawdown_from_initial_wealth(agent_returns)
    baseline_dd = drawdown_from_initial_wealth(baseline_returns)
    budget = np.maximum(floor, margin * baseline_dd)
    cost = np.square(np.maximum(0.0, agent_dd - budget)) / scale
    alpha = np.square(alpha_budget_ratio * budget) / scale
    return {
        "cost": float(np.mean(cost)),
        "alpha": float(np.mean(alpha)),
        "feasible": float(np.mean(cost) <= np.mean(alpha) + 1e-12),
        "budget_floor_fraction": float(np.mean(margin * baseline_dd <= floor)),
        "mean_drawdown_gap": float(np.mean(agent_dd - budget)),
    }


def drawdown_from_initial_wealth(returns: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return (peak - wealth) / peak


def own_drawdown_metrics(config, result: dict[str, object]) -> dict[str, float]:
    alpha = (
        float(config.rcpo.alpha_budget_ratio) ** 2
        * float(result["average_effective_drawdown_budget_squared"])
        / float(config.environment.drawdown_cost_scale)
    )
    cost = float(result["average_drawdown_constraint_cost"])
    return {"cost": cost, "alpha": alpha, "feasible": float(cost <= alpha + 1e-12)}


def write_plots(
    output_dir: Path,
    path_data: dict[str, list[np.ndarray]],
    rows: list[dict[str, object]],
    baseline_max_drawdowns: list[float],
    baseline_feasibility: list[float],
) -> None:
    for group_name, configs in GROUPS.items():
        labels = [CONDITIONS[name] for name in configs if CONDITIONS[name] in path_data]
        if not labels:
            continue
        plt.figure(figsize=(10, 6))
        for label in labels:
            mean_path = np.mean(np.stack(path_data[label]), axis=0)
            plt.plot(np.arange(1, len(mean_path) + 1), mean_path * 100, label=label)
        plt.axhline(0, color="black", linestyle="--", label="Constrained-neutral baseline")
        plt.xlabel("Future market step")
        plt.ylabel("Relative wealth vs baseline (%)")
        plt.title(f"V3.4 {group_name}: shared future markets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{group_name}_relative_wealth.png", dpi=160)
        plt.close()

        means = []
        feasible = []
        for label in labels:
            selected = [
                row for row in rows
                if row["condition"] == label and row["checkpoint"] == "checkpoint_best_return.pt"
            ]
            means.append(float(np.mean([row["max_drawdown"] for row in selected])))
            feasible.append(float(np.mean([row["common_feasible"] for row in selected])))
        risk_labels = [*labels, "Constrained-neutral baseline"]
        means.append(float(np.mean(baseline_max_drawdowns)))
        feasible.append(float(np.mean(baseline_feasibility)))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(risk_labels))
        ax.bar(x, np.asarray(means) * 100, color="steelblue")
        ax.set_ylabel("Mean maximum drawdown (%)")
        ax.set_xticks(x, risk_labels, rotation=20, ha="right")
        second = ax.twinx()
        second.plot(x, np.asarray(feasible) * 100, "o-", color="darkred")
        second.set_ylabel("Common-budget feasible branches (%)")
        second.set_ylim(0, 100)
        ax.set_title(f"V3.4 {group_name}: return-risk comparison")
        fig.tight_layout()
        fig.savefig(output_dir / f"{group_name}_risk.png", dpi=160)
        plt.close(fig)


def compare(
    specs: list[RunSpec],
    output_dir: Path,
    *,
    anchor_seeds: list[int],
    branch_count: int,
    evaluation_steps: int = 252,
) -> dict[str, object]:
    if branch_count < 1 or not anchor_seeds or evaluation_steps < 1:
        raise ValueError("At least one anchor and one branch are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_config(CONFIG_DIR / "base.yaml")
    anchor_config = replace(reference.market, lookback=20)
    loaded = []
    for spec in specs:
        for checkpoint in ("checkpoint_best_return.pt", "checkpoint_best_feasible.pt"):
            if not (spec.run_dir / checkpoint).exists():
                if checkpoint == "checkpoint_best_return.pt":
                    raise FileNotFoundError(spec.run_dir / checkpoint)
                continue
            config, metadata, _environments, policy_fn = load_policy(
                PolicySpec(label=spec.task_id, run_dir=spec.run_dir, checkpoint=checkpoint)
            )
            loaded.append((spec, checkpoint, config, metadata, policy_fn))

    all_rows: list[dict[str, object]] = []
    path_data: dict[str, list[np.ndarray]] = defaultdict(list)
    baseline_max_drawdowns: list[float] = []
    baseline_feasibility: list[float] = []
    for anchor_index, anchor_seed in enumerate(anchor_seeds):
        anchor = generate_market_split(anchor_config, reference.evaluation.fixed_anchor_steps, anchor_seed)
        for branch_index in range(branch_count):
            future_seed = 300_000 + 10_000 * anchor_index + 101 * branch_index
            baseline_market = generate_continuation_split(
                reference.market, anchor, evaluation_steps, future_seed
            )
            baseline_env = PortfolioEnv(
                replace(reference.environment, episode_length=evaluation_steps),
                baseline_market,
                reference.market,
            )
            baseline = rollout_policy(
                baseline_env,
                lambda _obs, env=baseline_env: env.constrained_neutral_action(),
            )
            baseline_returns = np.asarray(baseline["returns"])
            if len(baseline_returns) != evaluation_steps:
                raise ValueError("Baseline evaluation did not run for the requested steps")
            baseline_max_drawdowns.append(float(np.max(drawdown_from_initial_wealth(baseline_returns))))
            baseline_feasibility.append(common_drawdown_metrics(
                baseline_returns, baseline_returns
            )["feasible"])
            for spec, checkpoint, config, _metadata, policy_fn in loaded:
                if config.environment.episode_length != evaluation_steps:
                    raise ValueError(f"{spec.task_id} episode length does not match evaluation steps")
                market = generate_continuation_split(config.market, anchor, evaluation_steps, future_seed)
                if not np.array_equal(
                    market.risky_returns[-evaluation_steps:],
                    baseline_market.risky_returns[-evaluation_steps:],
                ):
                    raise ValueError(f"Future returns differ for {spec.task_id}")
                env = PortfolioEnv(config.environment, market, config.market)
                result = rollout_policy(env, policy_fn)
                policy_returns = np.asarray(result["returns"])
                if len(policy_returns) != evaluation_steps:
                    raise ValueError(f"{spec.task_id} did not run for the requested steps")
                common = common_drawdown_metrics(policy_returns, baseline_returns)
                own = own_drawdown_metrics(config, result)
                baseline_drawdown = drawdown_from_initial_wealth(baseline_returns)
                own_floor_fraction = float(np.mean(
                    config.environment.benchmark_drawdown_margin * baseline_drawdown
                    <= config.environment.drawdown_budget_floor
                ))
                label = CONDITIONS[spec.config_name]
                relative_path = relative_wealth_path(policy_returns, baseline_returns)
                if checkpoint == "checkpoint_best_return.pt":
                    path_data[label].append(relative_path)
                all_rows.append({
                    "task_id": spec.task_id,
                    "condition": label,
                    "seed": spec.seed,
                    "run_dir": str(spec.run_dir),
                    "checkpoint": checkpoint,
                    "anchor_seed": anchor_seed,
                    "branch_index": branch_index,
                    "final_cumulative_return": float(result["final_cumulative_return"]),
                    "baseline_final_cumulative_return": float(baseline["final_cumulative_return"]),
                    "relative_wealth": float(relative_path[-1]),
                    "max_drawdown": float(np.max(drawdown_from_initial_wealth(policy_returns))),
                    "average_turnover": float(result["average_turnover"]),
                    "own_cost": own["cost"],
                    "own_alpha": own["alpha"],
                    "own_feasible": own["feasible"],
                    "common_cost": common["cost"],
                    "common_alpha": common["alpha"],
                    "common_feasible": common["feasible"],
                    "common_budget_floor_fraction": common["budget_floor_fraction"],
                    "own_budget_floor_fraction": own_floor_fraction,
                    "common_mean_drawdown_gap": common["mean_drawdown_gap"],
                    "allocation_raw_cost": float(result["average_allocation_constraint_raw_cost"]),
                })

    csv_path = output_dir / "branch_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summaries = {}
    for label in sorted({str(row["condition"]) for row in all_rows}):
        summaries[label] = {}
        for checkpoint in sorted({str(row["checkpoint"]) for row in all_rows if row["condition"] == label}):
            selected = [row for row in all_rows if row["condition"] == label and row["checkpoint"] == checkpoint]
            per_seed = {}
            for seed in sorted({int(row["seed"]) for row in selected}):
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                per_seed[str(seed)] = {
                    key: float(np.mean([float(row[key]) for row in seed_rows]))
                    for key in ("relative_wealth", "max_drawdown", "average_turnover", "common_feasible", "own_feasible")
                }
            summaries[label][checkpoint] = {
                "seeds": per_seed,
                "mean_relative_wealth_across_seeds": float(np.mean([v["relative_wealth"] for v in per_seed.values()])),
                "std_relative_wealth_across_seeds": float(np.std([v["relative_wealth"] for v in per_seed.values()])),
                "common_feasible_branch_rate": float(np.mean([float(row["common_feasible"]) for row in selected])),
                "own_feasible_branch_rate": float(np.mean([float(row["own_feasible"]) for row in selected])),
            }
    comparisons = {}
    pairs = [
        ("market_count_2_to_8_e3", "Standalone 8x5040 e3", "Standalone 2x5040 e3"),
        ("market_count_2_to_8_e4", "Standalone 8x5040 e4", "Standalone 2x5040 e4"),
        ("epochs_3_to_4_m2", "Standalone 2x5040 e4", "Standalone 2x5040 e3"),
        ("epochs_3_to_4_m8", "Standalone 8x5040 e4", "Standalone 8x5040 e3"),
        ("counterfactual_e3", "Counterfactual 2x5040 e3", "Standalone 2x5040 e3"),
        ("counterfactual_e4", "Counterfactual 2x5040 e4", "Standalone 2x5040 e4"),
        *[(f"window_{n}", f"Window {n}", "Window 20") for n in (30, 40, 60)],
        *[(f"margin_{n}", f"Margin {n}", "Standalone 2x5040 e4") for n in ("0.85", "0.80")],
    ]
    for key, treatment, control in pairs:
        def keyed(label: str) -> dict[tuple[int, int, int], dict[str, object]]:
            return {
                (int(row["seed"]), int(row["anchor_seed"]), int(row["branch_index"])): row
                for row in all_rows
                if row["condition"] == label and row["checkpoint"] == "checkpoint_best_return.pt"
            }
        treatment_rows, control_rows = keyed(treatment), keyed(control)
        shared = sorted(treatment_rows.keys() & control_rows.keys())
        if not shared:
            continue
        deltas = np.asarray([
            float(treatment_rows[index]["relative_wealth"])
            - float(control_rows[index]["relative_wealth"])
            for index in shared
        ])
        seed_means = {
            str(seed): float(np.mean([
                delta for index, delta in zip(shared, deltas, strict=True)
                if index[0] == seed
            ]))
            for seed in sorted({index[0] for index in shared})
        }
        comparisons[key] = {
            "treatment": treatment,
            "control": control,
            "paired_branch_count": len(shared),
            "mean_relative_wealth_delta": float(np.mean(deltas)),
            "branch_win_rate": float(np.mean(deltas > 0)),
            "mean_delta_by_seed": seed_means,
        }
    write_plots(output_dir, path_data, all_rows, baseline_max_drawdowns, baseline_feasibility)
    summary = {
        "anchor_seeds": anchor_seeds,
        "branches_per_anchor": branch_count,
        "evaluation_steps": evaluation_steps,
        "common_drawdown_reference": {"margin": 0.90, "floor": 0.05, "scale": 0.10, "alpha_budget_ratio": 0.05},
        "constrained_neutral_baseline": {
            "mean_max_drawdown": float(np.mean(baseline_max_drawdowns)),
            "common_feasible_branch_rate": float(np.mean(baseline_feasibility)),
        },
        "conditions": summaries,
        "paired_comparisons": comparisons,
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired V3.4 comparison on shared future markets.")
    parser.add_argument("--runs-root", default="runs/v3.4")
    parser.add_argument("--output-dir", default="evaluation/v3.4_policy_comparison")
    parser.add_argument("--anchor-seeds", type=int, nargs="+", default=[91_000, 92_000, 93_000])
    parser.add_argument("--branches-per-anchor", type=int, default=20)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    specs = manifest_specs(Path(args.runs_root), controls_only=args.controls_only)
    summary = compare(specs, Path(args.output_dir), anchor_seeds=args.anchor_seeds, branch_count=args.branches_per_anchor)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
