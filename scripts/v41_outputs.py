"""Render V4.1 paper and Section 9 outputs from one shared set of rollouts."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter

from evaluate_paper import write_results
from compare_section9_simplex import load_validation_history, plot_bar_with_points, plot_validation_history
from rcpo_portfolio.evaluation import apply_signed_percent_axis, plt
from rcpo_portfolio.paper_evaluation import (
    BASELINE, CREDIT_METHODS, METHODS, write_csv,
)

COLORS = ("#252525", "#98733F", "#587DA3", "#A44638", "#218175", "#75549A", "#B96535", "#577F42")
SHORT = dict(zip(METHODS, ("Constrained-neutral", "Soft RCPO", "CAOSD PPO", "Shared", "Standalone", "CF-R", "CF-C", "CF-RC")))
PALETTE = {SHORT[method]: color for method, color in zip(METHODS, COLORS)}


def section9(folder, rows, learning, selection, seed):
    test = [r for r in rows if r["cohort"] == "main" and r["split"] == "test"]
    grouped = {m: [r for r in test if r["algorithm_id"] == m] for m in METHODS}
    summary = {}
    tag = "Seeds 0–2" if seed is None else f"Seed {seed}"
    checkpoint_label = "Best Return" if selection == "best_return" else "Feasibility-Selected"
    title = f"V4.1 {checkpoint_label} | {tag}"
    wealth = defaultdict(list)
    for method, records in grouped.items():
        summary[method] = {
            "display_name": METHODS[method],
            "mean_relative_wealth_vs_constrained_neutral": float(np.mean([r["relative_wealth"] for r in records])),
            "mean_final_cumulative_return": float(np.mean([r["actual_cumulative_return"] for r in records])),
            "mean_max_drawdown": float(np.mean([r["max_drawdown"] for r in records])),
            "mean_average_turnover": float(np.mean([r["average_turnover"] for r in records])),
            "mean_concentration": float(np.mean([r["mean_concentration"] for r in records])),
            "common_feasible_branch_rate": float(np.mean([r["path_feasible"] for r in records])),
            "mean_cost_gap": float(np.mean([r["cost_gap"] for r in records])),
        }
        if method == BASELINE:
            continue
        for r in records:
            with np.load(folder / r["trajectory"]) as trace:
                wealth[method].append(trace["wealth"] / trace["baseline_wealth"] - 1)
    for show_paths, name in (
        (True, "section9_cumulative_return_comparison.png"),
        (False, "section9_cumulative_return_means_only.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, color in zip(METHODS, COLORS):
            if method == BASELINE:
                ax.axhline(0, color=color, linestyle="--", label=SHORT[method])
                continue
            paths = np.asarray(wealth[method])
            if show_paths:
                ax.plot(paths.T, color=color, alpha=0.025, linewidth=0.5)
            ax.plot(paths.mean(axis=0), color=color, linewidth=2, label=SHORT[method])
        ax.set(title=f"{title} | Holdout Relative Wealth",
               xlabel="Trading step", ylabel="Relative wealth vs constrained-neutral")
        apply_signed_percent_axis(ax)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(folder / name, dpi=160)
        plt.close(fig)

    for file, metric, label, scale in (
        ("max_drawdown", "max_drawdown", "Maximum drawdown (%)", 100),
        ("turnover", "average_turnover", "Average L1 turnover (%)", 100),
        ("concentration", "mean_concentration", "Mean sum of squared weights", 1),
        ("common_drawdown_feasibility", "path_feasible", "Holdout paths meeting common budget (%)", 100),
    ):
        suffix = "" if file == "common_drawdown_feasibility" else "_comparison"
        plot_bar_with_points(
            folder / f"section9_{file}{suffix}.png", f"{title} | {label}", label,
            {SHORT[m]: [scale * r[metric] for r in records] for m, records in grouped.items()},
            colors=PALETTE,
        )

    histories = defaultdict(list)
    for method, update, relative_wealth in learning:
        histories[SHORT[method]].append({
            "update": update, "validation_relative_wealth": relative_wealth,
        })
    plot_validation_history(
        folder / "section9_validation_score_history.png", histories,
        colors=PALETTE, title=f"V4.1 | {tag} | Logged Training Validation",
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    feasible = [rec for rec in summary.values() if rec["mean_cost_gap"] <= 1e-10]
    frontier = sorted(
        (rec for rec in feasible if not any(
            other["mean_max_drawdown"] <= rec["mean_max_drawdown"]
            and other["mean_relative_wealth_vs_constrained_neutral"] >= rec["mean_relative_wealth_vs_constrained_neutral"]
            and (other["mean_max_drawdown"] < rec["mean_max_drawdown"]
                 or other["mean_relative_wealth_vs_constrained_neutral"] > rec["mean_relative_wealth_vs_constrained_neutral"])
            for other in feasible
        )),
        key=lambda row: row["mean_max_drawdown"],
    )
    if len(frontier) > 1:
        ax.plot(
            [r["mean_max_drawdown"] for r in frontier],
            [r["mean_relative_wealth_vs_constrained_neutral"] for r in frontier],
            ls="--", lw=1, color="#777777", label="Frontier (mean cost gap ≤ 0)",
        )
    for method, color in zip(METHODS, COLORS):
        rec = summary[method]
        rate = rec["common_feasible_branch_rate"]
        ax.scatter(rec["mean_max_drawdown"], rec["mean_relative_wealth_vs_constrained_neutral"],
                   s=65 + 300 * rate, c=color, edgecolor="white", linewidth=0.7,
                   label=f"{SHORT[method]} ({rate:.0%} feasible)")
    ax.axhline(0, color="#777777", lw=0.8)
    apply_signed_percent_axis(ax)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set(title=f"{title} | Return, Drawdown and Feasibility",
           xlabel="Mean maximum drawdown", ylabel="Mean terminal relative wealth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "section9_return_risk_feasibility_pareto.png", dpi=160)
    plt.close(fig)
    write_csv(folder / "section9_branch_metrics.csv", test)
    (folder / "section9_comparison_summary.json").write_text(json.dumps({
        "checkpoint_selection": selection, "training_seed": seed,
        "holdout_market_count": len(grouped[BASELINE]), "baseline": BASELINE,
        "feasibility": "path cost gap under the common evaluation rule",
        "policies": summary,
    }, indent=2, allow_nan=False), encoding="utf-8")


def paper_tables(folder, selection, seed):
    """Provide reading-ready tables alongside the unrounded CSVs."""
    def read(name):
        with (folder / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def format_number(value, scale=1, signed=False):
        if value in ("", None):
            return "N/A"
        return f"{float(value) * scale:+.2f}" if signed else f"{float(value) * scale:.2f}"

    def interval(row, metric, scale, signed=False):
        mean = format_number(row[f"mean_{metric}"], scale, signed)
        low = format_number(row[f"ci_low_{metric}"], scale, signed)
        high = format_number(row[f"ci_high_{metric}"], scale, signed)
        return f"{mean} [{low}, {high}]" if low != "N/A" else mean

    lines = [
        f"# V4.1 Paper Tables ({selection}, {'seeds 0–2' if seed is None else f'seed {seed}'})",
        "",
        "All test results use the same 100 held-out markets and the common drawdown rule.",
        "Intervals resample the available training seeds and holdout anchors. Single-seed results reflect market uncertainty.",
        "",
        "## Table 1. Overall Out-of-Sample Performance",
        "",
        "| Method | Relative wealth (%) [95% CI] | Actual cumulative return (%) [95% CI] | Mean cost gap (×10⁻³) [95% CI] | Path feasibility (%) | MDD (%) | Turnover (%) | Allocation violation steps (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in read("table_1_main_results.csv"):
        cells = (
            row["display_name"], interval(row, "relative_wealth", 100, True),
            interval(row, "actual_cumulative_return", 100), interval(row, "cost_gap", 1000, True),
            *(format_number(row[key], 100) for key in (
                "path_feasible_rate", "mean_max_drawdown", "mean_average_turnover", "allocation_violation_step_rate",
            )),
        )
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "", "## Table 2. Matched Credit Comparison Against Standalone", "",
        "| Method | Δ relative wealth (pp) [95% CI] | Δ mean cost gap (×10⁻³) [95% CI] |",
        "| --- | ---: | ---: |",
    ])
    for row in read("table_2_matched_credit.csv"):
        lines.append("| " + " | ".join((
            METHODS[row["algorithm_id"]], interval(row, "delta_relative_wealth", 100, True),
            interval(row, "delta_cost_gap", 1000, True),
        )) + " |")

    lines.extend([
        "", "## Table 3. Training and Deployment Costs", "",
        "| Method | Training transitions (M) | Training hours | Peak RSS (GiB) | Decision latency (ms) | Relative time | Relative memory | Relative latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in read("table_3_compute.csv"):
        columns = (
            ("training_real_transition_count", 1e-6), ("training_wall_time_seconds", 1/3600),
            ("peak_training_memory_bytes", 2**-30), ("decision_latency_ms", 1),
            ("relative_training_time", 1), ("relative_peak_memory", 1), ("relative_decision_latency", 1),
        )
        cells = (METHODS[row["algorithm_id"]], *(format_number(row[key], scale) for key, scale in columns))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "Training timing and memory ratios require matching hardware, timing boundaries, and interaction budgets.",
        "", "## Figure 1. Validation Learning Curves", "",
        "![Validation relative wealth and path-cost gap during training](figure_1_credit_learning.png)", "",
    ])
    (folder / "paper_tables.md").write_text("\n".join(lines), encoding="utf-8")


def write_v41_outputs(
    root, rows, alternative, figure_raw, matched, protocol, family, latency,
    counts, compute, bootstrap, bootstrap_seed,
):
    write_csv(root / "evaluation_episodes.csv", rows)
    training_history = [r for r in rows if r["split"] == "validation"]
    manifest_path = root.with_name(f"{root.name}_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logged_history = [
        (entry["algorithm_id"], entry["training_seed"],
         load_validation_history(manifest_path.parent / entry["run_dir"]))
        for entry in manifest["runs"]
    ]
    baseline = [r for r in rows if r["split"] == "test" and r["algorithm_id"] == BASELINE]
    best_return = [
        r for r in rows if r["split"] == "test" and r["cohort"] == "main"
        and r["algorithm_id"] != BASELINE
    ]
    for selection, selected in (("best_return", best_return), ("best_feasible", alternative)):
        for seed in (None, 0, 1, 2):
            folder = root / selection / ("average_seeds" if seed is None else f"seed{seed}")
            folder.mkdir(parents=True)
            def is_in_seed(record):
                return seed is None or record["training_seed"] in (None, seed)

            filtered = [
                {
                    **r, "cohort": "main",
                    "trajectory": Path(os.path.relpath(root / r["trajectory"], folder)).as_posix(),
                }
                for r in [*baseline, *selected, *training_history] if is_in_seed(r)
            ]
            paired = {
                method: [r for r in filtered if r["split"] == "test" and r["algorithm_id"] == method]
                for method in CREDIT_METHODS
            }
            figure = [point for point in figure_raw if is_in_seed(point[0])]
            scoped_compute = [item for item in compute if is_in_seed(item)]
            scoped_latency = defaultdict(list)
            target_cohort = "main" if selection == "best_return" else selection
            for (cohort, run_seed, method, device), samples in latency.items():
                if cohort in (None, target_cohort) and (seed is None or run_seed in (None, seed)):
                    scoped_latency[method, device].extend(samples)
            write_results(
                folder, filtered, figure, paired, protocol, family, scoped_latency, counts,
                scoped_compute, bootstrap, bootstrap_seed,
            )
            by_point = defaultdict(list)
            for method, run_seed, points in logged_history:
                if seed is None or seed == run_seed:
                    for point in points:
                        by_point[method, point["update"] + 1].append(point["validation_relative_wealth"])
            learning = [
                (method, update, float(np.mean(values)))
                for (method, update), values in sorted(by_point.items())
            ]
            section9(folder, filtered, learning, selection, seed)
            paper_tables(folder, selection, seed)
            print(f"[v4.1] wrote {folder}", flush=True)
    (root / "README.md").write_text(
        "# V4.1 Evaluation\n\n"
        "Each checkpoint selection contains one three-seed summary and three individual seed folders.\n"
        "Each folder contains the Section 9 plots and the paper tables and learning figure.\n\n"
        "- [Best return, seeds 0–2](best_return/average_seeds/paper_tables.md)\n"
        "- [Feasibility-selected, seeds 0–2](best_feasible/average_seeds/paper_tables.md)\n\n"
        "Both selections use the same 100 holdout markets. Figure 1 re-evaluates 31 snapshots per run "
        "on a fixed set of 20 markets. Section 9 validation history uses the original training logs. "
        "Per-step trajectory files are shared across folders.\n",
        encoding="utf-8",
    )
