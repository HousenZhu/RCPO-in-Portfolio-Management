from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_section9_simplex import (
    V26_POLICIES,
    V32_POLICIES,
    V33_POLICIES,
    annualized_return,
    annualized_volatility,
    cumulative_return_path,
    load_validation_history,
    rollout_policy,
)
from compare_v34 import common_drawdown_metrics, drawdown_from_initial_wealth
from rcpo_portfolio.config import load_config
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.evaluation import relative_wealth_path
from rcpo_portfolio.evaluation import compute_drawdown, load_checkpoint_for_evaluation
from rcpo_portfolio.market import generate_continuation_split, generate_market_split


OUTPUT_DEFAULT = ROOT / "evaluation" / "v3.3_v3.4_professor_meeting"
REPORT_DEFAULT = ROOT / "reports" / "v3_3_v3_4_evaluation_meeting.md"
ANCHOR_SEEDS_DEFAULT = [91_000, 92_000, 93_000, 94_000, 95_000]
FUTURE_SEED_BASE = 300_000
CACHE_VERSION = 1
CHECKPOINTS = (
    ("best_return", "checkpoint_best_return.pt"),
    ("robust_feasible", "checkpoint_best_feasible.pt"),
)

V34_RUNS = {
    "simplex_v3.4_standalone_m2_s5040_e3": (
        "v34_standalone_m2_e3", "Standalone 2x5040 e3", "factorial"
    ),
    "simplex_v3.4_standalone_m2_s5040_e4": (
        "v34_standalone_m2_e4", "Standalone 2x5040 e4", "factorial"
    ),
    "simplex_v3.4_standalone_m8_s5040_e3": (
        "v34_standalone_m8_e3", "Standalone 8x5040 e3", "factorial"
    ),
    "simplex_v3.4_standalone_m8_s5040_e4": (
        "v34_standalone_m8_e4", "Standalone 8x5040 e4", "factorial"
    ),
    "simplex_v3.4_cf_reward_m2_s5040_e3": (
        "v34_cf_reward_m2_e3", "Counterfactual 2x5040 e3", "counterfactual"
    ),
    "simplex_v3.4_cf_reward_m2_s5040_e4": (
        "v34_cf_reward_m2_e4", "Counterfactual 2x5040 e4", "counterfactual"
    ),
    "simplex_v3.4_window_20": ("v34_window_20", "Window 20", "window"),
    "simplex_v3.4_window_30": ("v34_window_30", "Window 30", "window"),
    "simplex_v3.4_window_40": ("v34_window_40", "Window 40", "window"),
    "simplex_v3.4_window_60": ("v34_window_60", "Window 60", "window"),
    "simplex_v3.4_margin_085": ("v34_margin_085", "Margin 0.85", "margin"),
    "simplex_v3.4_margin_080": ("v34_margin_080", "Margin 0.80", "margin"),
}

V33_CONDITIONS = {
    "RCPO Gaussian 8x5040 e3": ("v33_gaussian_m8_s5040_e3", "Gaussian 8x5040 e3", "market"),
    "RCPO Gaussian 2x5040 e3": ("v33_gaussian_m2_s5040_e3", "Gaussian 2x5040 e3", "market"),
    "RCPO Gaussian 8x10080 e3": ("v33_gaussian_m8_s10080_e3", "Gaussian 8x10080 e3", "market"),
    "RCPO Gaussian 2x10080 e3": ("v33_gaussian_m2_s10080_e3", "Gaussian 2x10080 e3", "market"),
    "RCPO Gaussian 2x20160 e3": ("v33_gaussian_m2_s20160_e3", "Gaussian 2x20160 e3", "market"),
    "RCPO Gaussian 2x10080 e4": ("v33_gaussian_m2_s10080_e4", "Gaussian RCPO e4", "method"),
    "PPO Gaussian 2x10080 e4": ("v33_ppo_gaussian_m2_s10080_e4", "Gaussian PPO e4", "method"),
    "RCPO Dirichlet 2x10080 e4": ("v33_dirichlet_m2_s10080_e4", "Dirichlet RCPO e4", "method"),
    "RCPO Allocation+Drawdown 2x10080 e3": (
        "v33_allocation_drawdown_m2_s10080_e3", "Allocation+drawdown RCPO", "method"
    ),
    "CF Reward Gaussian 2x10080 e3": (
        "v33_cf_reward_gaussian_m2_s10080_e3", "CF reward Gaussian", "method"
    ),
    "CF Reward Dirichlet 2x10080 e4": (
        "v33_cf_reward_dirichlet_m2_s10080_e4", "CF reward Dirichlet", "method"
    ),
}

COLORS = {
    "Standalone 2x5040 e3": "#4e79a7",
    "Standalone 2x5040 e4": "#f28e2b",
    "Standalone 8x5040 e3": "#59a14f",
    "Standalone 8x5040 e4": "#e15759",
    "Counterfactual 2x5040 e3": "#76b7b2",
    "Counterfactual 2x5040 e4": "#b07aa1",
    "V2.6 Gaussian standalone": "#777777",
    "V3.2 CF reward Gaussian": "#9467bd",
    "V3.3 market control": "#2ca02c",
    "V3.4 replicated standalone": "#d62728",
}


@dataclass(frozen=True)
class ExperimentSpec:
    spec_id: str
    version: str
    condition_id: str
    condition: str
    group: str
    seed: int
    run_dir: Path


class RolloutCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rollouts ("
            "cache_key TEXT PRIMARY KEY, returns BLOB NOT NULL, metrics_json TEXT NOT NULL)"
        )

    def get(self, key: str) -> tuple[np.ndarray, dict[str, float]] | None:
        row = self.connection.execute(
            "SELECT returns, metrics_json FROM rollouts WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        returns = np.frombuffer(row[0], dtype=np.float32).copy()
        return returns, json.loads(row[1])

    def put(self, key: str, returns: np.ndarray, metrics: dict[str, float]) -> None:
        values = np.asarray(returns, dtype=np.float32)
        self.connection.execute(
            "INSERT OR REPLACE INTO rollouts(cache_key, returns, metrics_json) VALUES (?, ?, ?)",
            (key, values.tobytes(), json.dumps(metrics, sort_keys=True)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _single_glob(pattern: str) -> Path:
    matches = sorted((ROOT / "runs").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one run matching {pattern}, found {len(matches)}")
    return matches[0]


def _prefix_match(name: str, mapping: dict[str, tuple[str, str, str]]) -> tuple[str, str, str]:
    matches = [value for prefix, value in mapping.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely classify run: {name}")
    return matches[0]


def discover_specs() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for item in V26_POLICIES:
        condition = item.label.replace(" v2.6", "")
        specs.append(ExperimentSpec(
            f"v26_{condition.lower().replace(' ', '_').replace('+', 'plus')}",
            "v2.6", f"v26_{condition.lower().replace(' ', '_')}", condition, "historical",
            0, item.run_dir,
        ))
    for item in V32_POLICIES:
        condition = item.label.replace(" v3.2", "")
        specs.append(ExperimentSpec(
            f"v32_{condition.lower().replace(' ', '_').replace('+', 'plus')}",
            "v3.2", f"v32_{condition.lower().replace(' ', '_')}", condition, "historical",
            0, item.run_dir,
        ))
    for item in V33_POLICIES:
        condition_id, condition, group = V33_CONDITIONS[item.label]
        specs.append(ExperimentSpec(
            condition_id, "v3.3", condition_id, condition, f"v33_{group}", 0, item.run_dir
        ))
    v34_roots = sorted((ROOT / "runs").glob("simplex_v3.4_*"))
    if len(v34_roots) != 12:
        raise ValueError(f"Expected 12 merged V3.4 run directories, found {len(v34_roots)}")
    new_count = 0
    for run_root in v34_roots:
        condition_id, condition, group = _prefix_match(run_root.name, V34_RUNS)
        for seed_dir in sorted(run_root.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            specs.append(ExperimentSpec(
                f"{condition_id}_s{seed}", "v3.4", condition_id, condition,
                f"v34_{group}", seed, seed_dir,
            ))
            new_count += 1
    if new_count != 22:
        raise ValueError(f"Expected 22 new V3.4 seed-runs, found {new_count}")
    controls = (
        (
            "v34_standalone_m2_e3", "Standalone 2x5040 e3", "v34_factorial",
            "simplex_v3.3_rcpo_gaussian_m2_s5040_e3_rcpo_none_*/seed_0",
        ),
        (
            "v34_standalone_m8_e3", "Standalone 8x5040 e3", "v34_factorial",
            "simplex_v3.3_rcpo_gaussian_m8_s5040_e3_rcpo_none_*/seed_0",
        ),
    )
    for condition_id, condition, group, pattern in controls:
        specs.append(ExperimentSpec(
            f"{condition_id}_s0_control", "v3.4", condition_id, condition, group, 0,
            _single_glob(pattern),
        ))
    validate_specs(specs)
    return specs


def validate_specs(specs: list[ExperimentSpec]) -> None:
    ids = [spec.spec_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment spec IDs")
    for spec in specs:
        summary_path = spec.run_dir / "training_summary.json"
        config_path = spec.run_dir / "config_snapshot.yaml"
        if not summary_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Incomplete run directory: {spec.run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if spec.version in {"v3.3", "v3.4"} and int(summary["completed_updates"]) != 60_000:
            raise ValueError(f"{spec.spec_id} did not complete 60,000 updates")
    v34 = [spec for spec in specs if spec.version == "v3.4"]
    expected = {
        "v34_standalone_m2_e3": {0, 1, 2},
        "v34_standalone_m2_e4": {0, 1, 2},
        "v34_standalone_m8_e3": {0, 1, 2},
        "v34_standalone_m8_e4": {0, 1, 2},
        "v34_cf_reward_m2_e3": {0, 1, 2},
        "v34_cf_reward_m2_e4": {0, 1, 2},
        "v34_window_20": {0}, "v34_window_30": {0},
        "v34_window_40": {0}, "v34_window_60": {0},
        "v34_margin_085": {0}, "v34_margin_080": {0},
    }
    actual = defaultdict(set)
    for spec in v34:
        actual[spec.condition_id].add(spec.seed)
    if dict(actual) != expected:
        raise ValueError(f"Unexpected V3.4 condition/seed matrix: {dict(actual)}")


def checkpoint_rows(specs: list[ExperimentSpec]) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        config = load_config(spec.run_dir / "config_snapshot.yaml")
        summary = json.loads((spec.run_dir / "training_summary.json").read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "spec_id": spec.spec_id, "version": spec.version,
            "condition_id": spec.condition_id, "condition": spec.condition,
            "group": spec.group, "seed": spec.seed,
            "run_dir": str(spec.run_dir), "completed_updates": summary["completed_updates"],
            "algorithm": summary["algo"], "policy_architecture": config.network.policy_architecture,
            "branch_credit_mode": config.network.branch_credit_mode,
            "train_market_count": config.market.train_market_count,
            "train_steps": config.market.train_steps, "lookback": config.market.lookback,
            "epochs": (config.ppo.epochs if summary["algo"] == "ppo_unconstrained" else config.optimization.epochs),
            "benchmark_drawdown_margin": config.environment.benchmark_drawdown_margin,
        }
        for checkpoint_type, filename in CHECKPOINTS:
            row[f"{checkpoint_type}_available"] = (spec.run_dir / filename).exists()
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def cache_key(*parts: object) -> str:
    payload = "|".join([str(CACHE_VERSION), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_metrics(result: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in result.items()
        if key != "returns" and np.isscalar(value)
    }


def batched_rollout(
    environments: list[PortfolioEnv],
    model,
    config,
) -> list[tuple[np.ndarray, dict[str, float]]]:
    observations = []
    histories = []
    for env in environments:
        observation, _ = env.reset(options={"start_index": int(env.available_start_indices()[0])})
        observations.append(observation)
        histories.append(defaultdict(list))
    device = next(model.parameters()).device
    active = np.ones(len(environments), dtype=bool)
    while np.any(active):
        tensor = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
        with torch.no_grad():
            output = model.get_policy_output(tensor, deterministic=config.evaluation.deterministic)
        actions = output.action.cpu().numpy()
        for index, env in enumerate(environments):
            if not active[index]:
                continue
            observation, _reward, terminated, truncated, info = env.step(actions[index])
            observations[index] = observation
            history = histories[index]
            history["returns"].append(float(info["net_return"]))
            history["turnover"].append(float(info["turnover"]))
            history["constraint"].append(float(info["constraint_cost"]))
            history["drawdown_cost"].append(float(info["drawdown_constraint_cost"]))
            history["allocation_1_weight"].append(float(info["allocation_constraint_1_weight"]))
            history["allocation_2_weight"].append(float(info["allocation_constraint_2_weight"]))
            history["allocation_1_violation"].append(float(info["allocation_constraint_1_violation_cost"]))
            history["allocation_2_violation"].append(float(info["allocation_constraint_2_violation_cost"]))
            history["allocation_raw"].append(float(info["allocation_constraint_raw_cost"]))
            history["allocation_cost"].append(float(info["allocation_constraint_cost"]))
            history["effective_budget"].append(float(info["effective_drawdown_budget"]))
            history["weights"].append(np.asarray(info["weights"], dtype=np.float32))
            if terminated or truncated:
                active[index] = False
    outputs = []
    for history in histories:
        returns = np.asarray(history["returns"], dtype=np.float32)
        weights = np.asarray(history["weights"], dtype=np.float32)
        concentration = np.sum(np.square(weights), axis=1)
        max_weight = np.max(weights, axis=1)
        metrics = {
            "final_cumulative_return": float(cumulative_return_path(returns)[-1]),
            "annualized_return": annualized_return(returns),
            "annualized_volatility": annualized_volatility(returns),
            "max_drawdown": float(np.max(compute_drawdown(returns))),
            "average_turnover": float(np.mean(history["turnover"])),
            "average_constraint_cost": float(np.mean(history["constraint"])),
            "average_drawdown_constraint_cost": float(np.mean(history["drawdown_cost"])),
            "average_allocation_constraint_1_weight": float(np.mean(history["allocation_1_weight"])),
            "average_allocation_constraint_2_weight": float(np.mean(history["allocation_2_weight"])),
            "average_allocation_constraint_1_violation_cost": float(np.mean(history["allocation_1_violation"])),
            "average_allocation_constraint_2_violation_cost": float(np.mean(history["allocation_2_violation"])),
            "average_allocation_constraint_raw_cost": float(np.mean(history["allocation_raw"])),
            "average_allocation_constraint_cost": float(np.mean(history["allocation_cost"])),
            "average_effective_drawdown_budget_squared": float(np.mean(np.square(history["effective_budget"]))),
            "average_concentration": float(np.mean(concentration)),
            "maximum_concentration": float(np.max(concentration)),
            "average_max_single_asset_weight": float(np.mean(max_weight)),
            "maximum_single_asset_weight": float(np.max(max_weight)),
            "fraction_steps_single_asset_above_70pct": float(np.mean(max_weight > 0.70)),
        }
        outputs.append((returns, metrics))
    return outputs


def own_alpha(config, result: dict[str, object]) -> float:
    if config.rcpo.alpha is not None:
        return float(config.rcpo.alpha)
    return (
        float(config.rcpo.alpha_budget_ratio) ** 2
        * float(result["average_effective_drawdown_budget_squared"])
        / float(config.environment.drawdown_cost_scale)
    )


def build_branches(reference_config, anchor_seeds: list[int], branch_count: int):
    canonical_market = replace(
        reference_config.market,
        lookback=60,
        train_generation_lookback=60,
    )
    branches = []
    for anchor_index, anchor_seed in enumerate(anchor_seeds):
        anchor = generate_market_split(
            canonical_market, reference_config.evaluation.fixed_anchor_steps, anchor_seed
        )
        for branch_index in range(branch_count):
            future_seed = FUTURE_SEED_BASE + 10_000 * anchor_index + 101 * branch_index
            canonical_future = generate_continuation_split(
                canonical_market, anchor, reference_config.environment.episode_length, future_seed
            )
            branches.append((anchor_index, anchor_seed, branch_index, future_seed, anchor, canonical_future))
    return branches


def evaluate(
    specs: list[ExperimentSpec], output_dir: Path, *, anchor_seeds: list[int], branch_count: int
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_config(ROOT / "configs" / "v3.4_experiment2_8assets" / "base.yaml")
    branches = build_branches(reference, anchor_seeds, branch_count)
    cache = RolloutCache(output_dir / "rollout_cache.sqlite")
    baseline: dict[tuple[int, int], tuple[np.ndarray, dict[str, float]]] = {}
    try:
        for anchor_index, anchor_seed, branch_index, future_seed, anchor, canonical in branches:
            key = cache_key("baseline", anchor_seed, future_seed, reference.environment.episode_length)
            cached = cache.get(key)
            if cached is None:
                market = generate_continuation_split(
                    reference.market, anchor, reference.environment.episode_length, future_seed
                )
                np.testing.assert_array_equal(
                    market.risky_returns[-reference.environment.episode_length :],
                    canonical.risky_returns[-reference.environment.episode_length :],
                )
                env = PortfolioEnv(reference.environment, market, reference.market)
                result = rollout_policy(env, lambda _obs, current=env: current.constrained_neutral_action())
                returns = np.asarray(result["returns"], dtype=np.float32)
                metrics = numeric_metrics(result)
                cache.put(key, returns, metrics)
                cached = returns, metrics
            baseline[(anchor_index, branch_index)] = cached

        rows: list[dict[str, object]] = []
        for spec_index, spec in enumerate(specs, start=1):
            for checkpoint_type, checkpoint_name in CHECKPOINTS:
                checkpoint_path = spec.run_dir / checkpoint_name
                if not checkpoint_path.exists():
                    continue
                print(
                    f"[meeting-eval] policy={spec_index}/{len(specs)} {spec.spec_id} "
                    f"checkpoint={checkpoint_type}", flush=True,
                )
                config, _metadata, model, _environments = load_checkpoint_for_evaluation(
                    spec.run_dir, checkpoint_name=checkpoint_name
                )
                cached_by_branch: dict[tuple[int, int], tuple[np.ndarray, dict[str, float]]] = {}
                missing_environments: list[PortfolioEnv] = []
                missing_keys: list[str] = []
                missing_branches: list[tuple[int, int]] = []
                for anchor_index, anchor_seed, branch_index, future_seed, anchor, canonical in branches:
                    key = cache_key(
                        "policy", checkpoint_signature(checkpoint_path), anchor_seed,
                        future_seed, config.environment.episode_length,
                    )
                    cached = cache.get(key)
                    if cached is None:
                        market = generate_continuation_split(
                            config.market, anchor, config.environment.episode_length, future_seed
                        )
                        np.testing.assert_array_equal(
                            market.risky_returns[-config.environment.episode_length :],
                            canonical.risky_returns[-config.environment.episode_length :],
                        )
                        missing_environments.append(
                            PortfolioEnv(config.environment, market, config.market)
                        )
                        missing_keys.append(key)
                        missing_branches.append((anchor_index, branch_index))
                    else:
                        cached_by_branch[(anchor_index, branch_index)] = cached

                if missing_environments:
                    print(
                        f"[meeting-eval] batching {len(missing_environments)} uncached markets",
                        flush=True,
                    )
                    outputs = batched_rollout(missing_environments, model, config)
                    for branch_key, key, (returns, metrics) in zip(
                        missing_branches, missing_keys, outputs, strict=True
                    ):
                        if len(returns) != 252:
                            raise ValueError(
                                f"{spec.spec_id} produced {len(returns)} evaluation steps"
                            )
                        cache.put(key, returns, metrics)
                        cached_by_branch[branch_key] = returns, metrics

                for anchor_index, anchor_seed, branch_index, future_seed, _anchor, _canonical in branches:
                    returns, metrics = cached_by_branch[(anchor_index, branch_index)]
                    baseline_returns, baseline_metrics = baseline[(anchor_index, branch_index)]
                    relative_path = relative_wealth_path(returns, baseline_returns)
                    common = common_drawdown_metrics(returns, baseline_returns)
                    alpha = own_alpha(config, metrics)
                    raw_allocation = float(metrics["average_allocation_constraint_raw_cost"])
                    if config.environment.action_mode == "simplex_decomposition" and raw_allocation > 1e-8:
                        raise ValueError(
                            f"Hard allocation violation in {spec.spec_id}: {raw_allocation}"
                        )
                    own_cost = float(metrics["average_constraint_cost"])
                    baseline_dd = drawdown_from_initial_wealth(baseline_returns)
                    own_floor_fraction = float(np.mean(
                        float(config.environment.benchmark_drawdown_margin) * baseline_dd
                        <= float(config.environment.drawdown_budget_floor)
                    ))
                    rows.append({
                        "spec_id": spec.spec_id, "version": spec.version,
                        "condition_id": spec.condition_id, "condition": spec.condition,
                        "group": spec.group, "seed": spec.seed,
                        "run_dir": str(spec.run_dir), "checkpoint": checkpoint_name,
                        "checkpoint_type": checkpoint_type, "anchor_index": anchor_index,
                        "anchor_seed": anchor_seed, "branch_index": branch_index,
                        "future_seed": future_seed,
                        "relative_wealth": float(relative_path[-1]),
                        "final_cumulative_return": float(metrics["final_cumulative_return"]),
                        "baseline_final_cumulative_return": float(baseline_metrics["final_cumulative_return"]),
                        "annualized_return": float(metrics["annualized_return"]),
                        "annualized_volatility": float(metrics["annualized_volatility"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "average_turnover": float(metrics["average_turnover"]),
                        "own_cost": own_cost, "own_alpha": alpha,
                        "own_feasible": float(own_cost <= alpha + 1e-12),
                        "common_cost": common["cost"], "common_alpha": common["alpha"],
                        "common_feasible": common["feasible"],
                        "common_budget_floor_fraction": common["budget_floor_fraction"],
                        "own_budget_floor_fraction": own_floor_fraction,
                        "common_mean_drawdown_gap": common["mean_drawdown_gap"],
                        "allocation_raw_cost": raw_allocation,
                        "average_concentration": float(metrics["average_concentration"]),
                    })
        write_csv(output_dir / "branch_metrics.csv", rows)
        return rows
    finally:
        cache.close()


def validation_relative(spec: ExperimentSpec, checkpoint_type: str) -> float | None:
    summary = json.loads((spec.run_dir / "training_summary.json").read_text(encoding="utf-8"))
    key = "best_validation" if checkpoint_type == "best_return" else "best_feasible_validation"
    payload = summary.get(key)
    if not payload:
        return None
    value = payload.get("mean_relative_wealth_vs_constrained_neutral")
    return None if value is None else float(value)


def anchor_bootstrap_ci(rows: list[dict[str, object]], field: str) -> tuple[float, float]:
    anchor_means = []
    for anchor in sorted({int(row["anchor_index"]) for row in rows}):
        values = [float(row[field]) for row in rows if int(row["anchor_index"]) == anchor]
        anchor_means.append(float(np.mean(values)))
    rng = np.random.default_rng(34)
    values = np.asarray(anchor_means)
    draws = rng.choice(values, size=(5_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_seeds(specs: list[ExperimentSpec], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {spec.spec_id: spec for spec in specs}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["spec_id"]), str(row["checkpoint_type"]))].append(row)
    summaries = []
    for (spec_id, checkpoint_type), selected in sorted(grouped.items()):
        relative = np.asarray([float(row["relative_wealth"]) for row in selected])
        q10 = float(np.quantile(relative, 0.10))
        tail = relative[relative <= q10 + 1e-15]
        low, high = anchor_bootstrap_ci(selected, "relative_wealth")
        spec = by_id[spec_id]
        val = validation_relative(spec, checkpoint_type)
        summaries.append({
            "spec_id": spec_id, "version": spec.version,
            "condition_id": spec.condition_id, "condition": spec.condition,
            "group": spec.group, "seed": spec.seed, "checkpoint_type": checkpoint_type,
            "market_count": len(selected), "mean_relative_wealth": float(np.mean(relative)),
            "ci_low_relative_wealth": low, "ci_high_relative_wealth": high,
            "win_rate": float(np.mean(relative > 0.0)),
            "q10_relative_wealth": q10, "cvar10_relative_wealth": float(np.mean(tail)),
            "mean_max_drawdown": float(np.mean([float(row["max_drawdown"]) for row in selected])),
            "mean_turnover": float(np.mean([float(row["average_turnover"]) for row in selected])),
            "common_feasible_rate": float(np.mean([float(row["common_feasible"]) for row in selected])),
            "own_feasible_rate": float(np.mean([float(row["own_feasible"]) for row in selected])),
            "common_floor_fraction": float(np.mean([float(row["common_budget_floor_fraction"]) for row in selected])),
            "own_floor_fraction": float(np.mean([float(row["own_budget_floor_fraction"]) for row in selected])),
            "validation_relative_wealth": val if val is not None else "",
            "validation_holdout_gap": (val - float(np.mean(relative))) if val is not None else "",
        })
    return summaries


PAIRINGS = (
    ("market_2_to_8_e3", "v34_standalone_m8_e3", "v34_standalone_m2_e3"),
    ("market_2_to_8_e4", "v34_standalone_m8_e4", "v34_standalone_m2_e4"),
    ("epochs_3_to_4_m2", "v34_standalone_m2_e4", "v34_standalone_m2_e3"),
    ("epochs_3_to_4_m8", "v34_standalone_m8_e4", "v34_standalone_m8_e3"),
    ("counterfactual_e3", "v34_cf_reward_m2_e3", "v34_standalone_m2_e3"),
    ("counterfactual_e4", "v34_cf_reward_m2_e4", "v34_standalone_m2_e4"),
    ("window_30_vs_20", "v34_window_30", "v34_window_20"),
    ("window_40_vs_20", "v34_window_40", "v34_window_20"),
    ("window_60_vs_20", "v34_window_60", "v34_window_20"),
    ("margin_085_vs_090", "v34_margin_085", "v34_standalone_m2_e4"),
    ("margin_080_vs_090", "v34_margin_080", "v34_standalone_m2_e4"),
)


def paired_effects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for name, treatment, control in PAIRINGS:
        for checkpoint_type, _filename in CHECKPOINTS:
            for seed in sorted({int(row["seed"]) for row in rows}):
                def keyed(condition_id: str):
                    return {
                        (int(row["anchor_index"]), int(row["branch_index"])): row
                        for row in rows
                        if row["condition_id"] == condition_id
                        and row["checkpoint_type"] == checkpoint_type
                        and int(row["seed"]) == seed
                    }
                left, right = keyed(treatment), keyed(control)
                shared = sorted(left.keys() & right.keys())
                if not shared:
                    continue
                rel = np.asarray([
                    float(left[index]["relative_wealth"]) - float(right[index]["relative_wealth"])
                    for index in shared
                ])
                output.append({
                    "comparison": name, "treatment": treatment, "control": control,
                    "checkpoint_type": checkpoint_type, "seed": seed,
                    "paired_markets": len(shared), "mean_relative_wealth_delta": float(np.mean(rel)),
                    "paired_win_rate": float(np.mean(rel > 0.0)),
                    "mean_max_drawdown_delta": float(np.mean([
                        float(left[index]["max_drawdown"]) - float(right[index]["max_drawdown"])
                        for index in shared
                    ])),
                    "common_feasible_rate_delta": float(np.mean([
                        float(left[index]["common_feasible"]) - float(right[index]["common_feasible"])
                        for index in shared
                    ])),
                    "mean_turnover_delta": float(np.mean([
                        float(left[index]["average_turnover"]) - float(right[index]["average_turnover"])
                        for index in shared
                    ])),
                })
    return output


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.22,
    })


def _percent_axis(axis, *, x: bool = False) -> None:
    from matplotlib.ticker import FuncFormatter
    limits = axis.get_xlim() if x else axis.get_ylim()
    decimals = 1 if abs(limits[1] - limits[0]) < 0.08 else 0
    formatter = FuncFormatter(
        lambda value, _position: (
            f"{value:+.{decimals}%}" if abs(value) > 1e-12 else f"{0:.{decimals}%}"
        )
    )
    (axis.xaxis if x else axis.yaxis).set_major_formatter(formatter)


def _seed_metric(
    summaries: list[dict[str, object]], condition_ids: list[str], checkpoint: str
) -> dict[str, list[dict[str, object]]]:
    grouped = defaultdict(list)
    for row in summaries:
        if row["condition_id"] in condition_ids and row["checkpoint_type"] == checkpoint:
            grouped[str(row["condition_id"])].append(row)
    return grouped


def _dot_panel(axis, grouped, labels, field, title, ylabel, *, percent=True) -> None:
    rng = np.random.default_rng(34)
    for index, condition_id in enumerate(labels):
        rows = grouped.get(condition_id, [])
        values = [float(row[field]) for row in rows]
        if not values:
            continue
        color = COLORS.get(grouped[condition_id][0]["condition"], "#4e79a7")
        positions = index + rng.normal(0, 0.035, len(values))
        if field == "mean_relative_wealth":
            axis.vlines(
                positions,
                [float(row["ci_low_relative_wealth"]) for row in rows],
                [float(row["ci_high_relative_wealth"]) for row in rows],
                color=color, alpha=0.35, linewidth=1.5, zorder=2,
            )
        axis.scatter(positions, values, s=48, color=color,
                     edgecolor="white", linewidth=0.7, zorder=3)
        axis.plot([index - 0.18, index + 0.18], [np.mean(values)] * 2, color="black", lw=2)
    axis.set_xticks(range(len(labels)), [grouped[label][0]["condition"] if grouped.get(label) else label for label in labels], rotation=18, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y")
    if percent:
        _percent_axis(axis)


def plot_v33(summaries, output_dir: Path) -> None:
    selected = [row for row in summaries if row["version"] == "v3.3" and row["checkpoint_type"] == "best_return"]
    market_ids = [
        "v33_gaussian_m2_s5040_e3", "v33_gaussian_m2_s10080_e3",
        "v33_gaussian_m2_s20160_e3", "v33_gaussian_m8_s5040_e3",
        "v33_gaussian_m8_s10080_e3",
    ]
    market = {str(row["condition_id"]): row for row in selected if row["condition_id"] in market_ids}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for count, color in ((2, "#4e79a7"), (8, "#e15759")):
        points = []
        for condition_id, row in market.items():
            if f"m{count}_" not in condition_id:
                continue
            steps = int(condition_id.split("_s")[1].split("_")[0])
            points.append((steps, row))
        points.sort()
        x = [p[0] for p in points]
        y = [float(p[1]["mean_relative_wealth"]) for p in points]
        axes[0].errorbar(
            x, y,
            yerr=[
                [value - float(point[1]["ci_low_relative_wealth"]) for value, point in zip(y, points, strict=True)],
                [float(point[1]["ci_high_relative_wealth"]) - value for value, point in zip(y, points, strict=True)],
            ],
            fmt="o-", capsize=3, color=color, alpha=0.9, label=f"{count} markets",
        )
        axes[1].plot([p[0] for p in points], [float(p[1]["win_rate"]) for p in points], "o-", color=color, label=f"{count} markets")
    axes[0].set_title("Holdout relative wealth")
    axes[0].set_ylabel("Relative wealth vs constrained-neutral")
    _percent_axis(axes[0])
    axes[1].set_title("Branch win rate")
    _percent_axis(axes[1])
    for axis in axes:
        axis.set_xlabel("Training steps per market")
        axis.grid(True)
        axis.legend()
    fig.suptitle("V3.3 Market Design (single-seed exploratory evidence)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "v33_01_market_design_return.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    ordered = [market[key] for key in market_ids if key in market]
    labels = [str(row["condition"]).replace("Gaussian ", "") for row in ordered]
    metrics = (("mean_max_drawdown", "Mean max drawdown"), ("common_feasible_rate", "Common feasible rate"), ("cvar10_relative_wealth", "Worst-10% relative wealth"))
    for axis, (field, title) in zip(axes, metrics, strict=True):
        axis.bar(range(len(ordered)), [float(row[field]) for row in ordered], color="#4e79a7")
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y")
        _percent_axis(axis)
    fig.suptitle("V3.3 Market Design Risk (single-seed)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "v33_02_market_design_risk.png")
    plt.close(fig)

    method_ids = [
        "v33_ppo_gaussian_m2_s10080_e4", "v33_gaussian_m2_s10080_e4",
        "v33_dirichlet_m2_s10080_e4", "v33_gaussian_m2_s10080_e3",
        "v33_allocation_drawdown_m2_s10080_e3", "v33_cf_reward_gaussian_m2_s10080_e3",
        "v33_cf_reward_dirichlet_m2_s10080_e4",
    ]
    methods = {str(row["condition_id"]): row for row in selected if row["condition_id"] in method_ids}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    panels = (method_ids[:3], method_ids[3:])
    titles = ("Optimizer and distribution", "Constraint and branch-credit variants")
    for axis, ids, title in zip(axes, panels, titles, strict=True):
        values = [methods[key] for key in ids if key in methods]
        y = np.arange(len(values))
        axis.scatter([float(row["mean_relative_wealth"]) for row in values], y, s=70, color="#4e79a7", label="Relative wealth")
        axis.scatter([float(row["common_feasible_rate"]) for row in values], y, s=70, marker="s", color="#e15759", label="Common feasible rate")
        axis.set_yticks(y, [str(row["condition"]) for row in values])
        axis.axvline(0, color="#777777", lw=0.8)
        axis.set_title(title)
        _percent_axis(axis, x=True)
        axis.grid(axis="x")
        axis.legend()
    fig.suptitle("V3.3 Method Ablation (single-seed exploratory evidence)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "v33_03_method_ablation.png")
    plt.close(fig)

    valid = [row for row in selected if row["validation_relative_wealth"] != ""]
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    x = np.asarray([float(row["validation_relative_wealth"]) for row in valid])
    y = np.asarray([float(row["mean_relative_wealth"]) for row in valid])
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    axis.plot([low, high], [low, high], "--", color="#777777", label="No generalization gap")
    axis.scatter(x, y, s=65, color="#4e79a7")
    for row, xv, yv in zip(valid, x, y, strict=True):
        axis.annotate(str(row["condition"]), (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    axis.set_xlabel("Selected validation relative wealth")
    axis.set_ylabel("Independent holdout relative wealth")
    axis.set_title("V3.3 Validation-to-Holdout Generalization Gap")
    _percent_axis(axis, x=True); _percent_axis(axis)
    axis.grid(True); axis.legend()
    fig.tight_layout(); fig.savefig(output_dir / "v33_04_validation_holdout_gap.png"); plt.close(fig)


def plot_v34(summaries, effects, output_dir: Path) -> None:
    factorial = ["v34_standalone_m2_e3", "v34_standalone_m2_e4", "v34_standalone_m8_e3", "v34_standalone_m8_e4"]
    for checkpoint, filename, title in (
        ("best_return", "v34_01_factorial_best_return.png", "Best-return checkpoint"),
        ("robust_feasible", "v34_02_factorial_robust_feasible.png", "Robust-feasible-selected checkpoint"),
    ):
        grouped = _seed_metric(summaries, factorial, checkpoint)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        _dot_panel(axes[0], grouped, factorial, "mean_relative_wealth", "Holdout return", "Relative wealth")
        _dot_panel(axes[1], grouped, factorial, "win_rate", "Branch win rate", "Win rate")
        fig.suptitle(f"V3.4 Replicated Factorial: {title}", fontweight="bold")
        fig.tight_layout(); fig.savefig(output_dir / filename); plt.close(fig)

    grouped = _seed_metric(summaries, factorial, "best_return")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    _dot_panel(axes[0], grouped, factorial, "mean_max_drawdown", "Maximum drawdown", "Max drawdown")
    _dot_panel(axes[1], grouped, factorial, "common_feasible_rate", "Common-rule feasibility", "Feasible branches")
    _dot_panel(axes[2], grouped, factorial, "cvar10_relative_wealth", "Worst-10% relative wealth", "CVaR10 relative wealth")
    fig.suptitle("V3.4 Factorial Risk: Best-return Checkpoints", fontweight="bold")
    fig.tight_layout(); fig.savefig(output_dir / "v34_03_factorial_risk.png"); plt.close(fig)

    grouped_robust = _seed_metric(summaries, factorial, "robust_feasible")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    _dot_panel(axes[0], grouped_robust, factorial, "mean_max_drawdown", "Maximum drawdown", "Max drawdown")
    _dot_panel(axes[1], grouped_robust, factorial, "common_feasible_rate", "Common-rule feasibility", "Feasible branches")
    _dot_panel(axes[2], grouped_robust, factorial, "cvar10_relative_wealth", "Worst-10% relative wealth", "CVaR10 relative wealth")
    fig.suptitle("V3.4 Factorial Risk: Robust-feasible-selected Checkpoints", fontweight="bold")
    fig.tight_layout(); fig.savefig(output_dir / "v34_03b_factorial_robust_feasible_risk.png"); plt.close(fig)

    effect_names = ["market_2_to_8_e3", "market_2_to_8_e4", "epochs_3_to_4_m2", "epochs_3_to_4_m8"]
    _plot_effects(effects, effect_names, output_dir / "v34_04_factorial_paired_effects.png", "V3.4 Factorial Paired Effects")

    credit_ids = ["v34_standalone_m2_e3", "v34_cf_reward_m2_e3", "v34_standalone_m2_e4", "v34_cf_reward_m2_e4"]
    grouped = _seed_metric(summaries, credit_ids, "best_return")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    _dot_panel(axes[0], grouped, credit_ids, "mean_relative_wealth", "Holdout return", "Relative wealth")
    _dot_panel(axes[1], grouped, credit_ids, "cvar10_relative_wealth", "Worst-10% outcome", "CVaR10 relative wealth")
    fig.suptitle("V3.4 Standalone vs Counterfactual Reward", fontweight="bold")
    fig.tight_layout(); fig.savefig(output_dir / "v34_05_credit_return.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    _dot_panel(axes[0], grouped, credit_ids, "mean_max_drawdown", "Maximum drawdown", "Max drawdown")
    _dot_panel(axes[1], grouped, credit_ids, "common_feasible_rate", "Common feasibility", "Feasible branches")
    _dot_panel(axes[2], grouped, credit_ids, "mean_turnover", "Turnover", "Average turnover", percent=False)
    fig.suptitle("V3.4 Credit Assignment Risk", fontweight="bold")
    fig.tight_layout(); fig.savefig(output_dir / "v34_06_credit_risk.png"); plt.close(fig)

    robust_credit = _seed_metric(summaries, credit_ids, "robust_feasible")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    _dot_panel(axes[0], robust_credit, credit_ids, "mean_relative_wealth", "Holdout return", "Relative wealth")
    _dot_panel(axes[1], robust_credit, credit_ids, "cvar10_relative_wealth", "Worst-10% outcome", "CVaR10 relative wealth")
    fig.suptitle("V3.4 Standalone vs Counterfactual: Robust-feasible-selected", fontweight="bold")
    fig.tight_layout(); fig.savefig(output_dir / "v34_05b_credit_robust_feasible.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    _dot_panel(axes[0], robust_credit, credit_ids, "mean_max_drawdown", "Maximum drawdown", "Max drawdown")
    _dot_panel(axes[1], robust_credit, credit_ids, "common_feasible_rate", "Common feasibility", "Feasible branches")
    _dot_panel(axes[2], robust_credit, credit_ids, "mean_turnover", "Turnover", "Average turnover", percent=False)
    fig.suptitle("V3.4 Credit Risk: Robust-feasible-selected", fontweight="bold")
    fig.tight_layout(); fig.savefig(output_dir / "v34_06b_credit_robust_feasible_risk.png"); plt.close(fig)
    _plot_effects(effects, ["counterfactual_e3", "counterfactual_e4"], output_dir / "v34_07_credit_paired_effects.png", "Counterfactual Reward Paired Effects")

    window_ids = ["v34_window_20", "v34_window_30", "v34_window_40", "v34_window_60"]
    margin_ids = ["v34_standalone_m2_e4", "v34_margin_085", "v34_margin_080"]
    _plot_sweep(summaries, window_ids, output_dir / "v34_08_window_sweep.png", "Observation Window (single-seed exploratory)", include_own=False)
    _plot_sweep(summaries, window_ids, output_dir / "v34_08b_window_robust_feasible.png", "Observation Window: Robust-feasible-selected (single seed)", include_own=False, checkpoint="robust_feasible")
    _plot_sweep(summaries, margin_ids, output_dir / "v34_09_margin_sweep.png", "Drawdown Margin (single-seed exploratory)", include_own=True)
    _plot_sweep(summaries, margin_ids, output_dir / "v34_09b_margin_robust_feasible.png", "Drawdown Margin: Robust-feasible-selected (single seed)", include_own=True, checkpoint="robust_feasible")


def _plot_effects(effects, names, path: Path, title: str) -> None:
    selected = [row for row in effects if row["comparison"] in names and row["checkpoint_type"] == "best_return"]
    fig, axis = plt.subplots(figsize=(10, 5.6))
    ylabels = []
    y = 0
    for name in names:
        rows = [row for row in selected if row["comparison"] == name]
        for row in rows:
            axis.scatter(float(row["mean_relative_wealth_delta"]), y, s=55, color="#4e79a7")
            ylabels.append(f"{name} / seed {row['seed']}")
            y += 1
    axis.axvline(0, color="#333333", linestyle="--")
    axis.set_yticks(range(len(ylabels)), ylabels)
    axis.set_xlabel("Paired relative-wealth effect (treatment - control)")
    axis.set_title(title)
    _percent_axis(axis, x=True); axis.grid(axis="x")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _plot_sweep(
    summaries, condition_ids, path: Path, title: str, *, include_own: bool,
    checkpoint: str = "best_return",
) -> None:
    selected = [
        row for row in summaries
        if row["condition_id"] in condition_ids and row["checkpoint_type"] == checkpoint
    ]
    by_id = {str(row["condition_id"]): row for row in selected}
    rows = [by_id[key] for key in condition_ids if key in by_id]
    fields = [
        ("mean_relative_wealth", "Relative wealth"),
        ("mean_max_drawdown", "Maximum drawdown"),
        (("own_feasible_rate" if include_own else "common_feasible_rate"), ("Own-rule feasibility" if include_own else "Common feasibility")),
        (("own_floor_fraction" if include_own else "mean_turnover"), ("Own budget floor active" if include_own else "Turnover")),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (field, subtitle) in zip(axes.flat, fields, strict=True):
        values = [float(row[field]) for row in rows]
        axis.plot(range(len(rows)), values, "o-", color="#4e79a7", lw=2)
        axis.set_xticks(range(len(rows)), [str(row["condition"]) for row in rows], rotation=15, ha="right")
        axis.set_title(subtitle); axis.grid(True)
        if field != "mean_turnover":
            _percent_axis(axis)
    if include_own:
        common = [float(row["common_feasible_rate"]) for row in rows]
        axes[1, 0].plot(range(len(rows)), common, "s--", color="#e15759", label="Common 0.90 rule")
        axes[1, 0].legend()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_overview(specs, summaries, rows, output_dir: Path) -> None:
    representatives = {
        "v26_rcpo_gaussian": "V2.6 Gaussian standalone",
        "v32_cf_reward_gaussian": "V3.2 CF reward Gaussian",
        "v33_gaussian_m2_s5040_e3": "V3.3 market control",
        "v34_standalone_m2_e4": "V3.4 replicated standalone",
    }
    selected = [
        row for row in summaries
        if row["checkpoint_type"] == "best_return" and row["condition_id"] in representatives
    ]
    fig, axis = plt.subplots(figsize=(8.5, 6.4))
    for condition_id, display in representatives.items():
        values = [row for row in selected if row["condition_id"] == condition_id]
        if not values:
            continue
        x = float(np.mean([float(row["mean_max_drawdown"]) for row in values]))
        y = float(np.mean([float(row["mean_relative_wealth"]) for row in values]))
        size = 120 + 260 * float(np.mean([float(row["common_feasible_rate"]) for row in values]))
        axis.scatter(x, y, s=size, color=COLORS[display], edgecolor="white", lw=1.2)
        axis.annotate(display, (x, y), xytext=(6, 5), textcoords="offset points")
    axis.axhline(0, color="#777777", linestyle="--")
    axis.set_xlabel("Mean maximum drawdown (lower is better)")
    axis.set_ylabel("Mean relative wealth vs constrained-neutral")
    axis.set_title("Cross-Version Return-Risk Evidence\nMarker size = common feasible-branch rate")
    _percent_axis(axis, x=True); _percent_axis(axis); axis.grid(True)
    fig.tight_layout(); fig.savefig(output_dir / "overview_01_cross_version_return_risk.png"); plt.close(fig)

    trade_ids = ["v32_cf_reward_gaussian", "v33_gaussian_m2_s5040_e3", "v34_standalone_m2_e4"]
    fig, axis = plt.subplots(figsize=(8.7, 6.4))
    for condition_id in trade_ids:
        by_checkpoint = {
            str(row["checkpoint_type"]): row for row in summaries if row["condition_id"] == condition_id
        }
        if not {"best_return", "robust_feasible"}.issubset(by_checkpoint):
            continue
        start, end = by_checkpoint["best_return"], by_checkpoint["robust_feasible"]
        x1, y1 = float(start["mean_max_drawdown"]), float(start["mean_relative_wealth"])
        x2, y2 = float(end["mean_max_drawdown"]), float(end["mean_relative_wealth"])
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
        axis.scatter([x1], [y1], marker="o", s=70)
        axis.scatter([x2], [y2], marker="s", s=70)
        axis.annotate(str(start["condition"]), (x1, y1), fontsize=8)
    axis.set_xlabel("Mean maximum drawdown")
    axis.set_ylabel("Mean relative wealth")
    axis.set_title("Checkpoint Trade-off: Best Return → Robust-Feasible-Selected")
    _percent_axis(axis, x=True); _percent_axis(axis); axis.grid(True)
    fig.tight_layout(); fig.savefig(output_dir / "overview_02_checkpoint_tradeoff.png"); plt.close(fig)

    paths: dict[str, list[np.ndarray]] = defaultdict(list)
    baseline_paths: list[np.ndarray] = []
    cache = RolloutCache(output_dir / "rollout_cache.sqlite")
    try:
        spec_by_id = {spec.spec_id: spec for spec in specs}
        branch_keys = sorted({(int(row["anchor_index"]), int(row["anchor_seed"]), int(row["branch_index"]), int(row["future_seed"])) for row in rows})
        reference = load_config(ROOT / "configs" / "v3.4_experiment2_8assets" / "base.yaml")
        for anchor_index, anchor_seed, branch_index, future_seed in branch_keys:
            base_key = cache_key("baseline", anchor_seed, future_seed, 252)
            base = cache.get(base_key)
            if base is None:
                continue
            baseline_returns = base[0]
            baseline_paths.append(np.zeros_like(baseline_returns))
            for condition_id, display in representatives.items():
                candidate_specs = [spec for spec in specs if spec.condition_id == condition_id]
                for spec in candidate_specs:
                    checkpoint = spec.run_dir / "checkpoint_best_return.pt"
                    if not checkpoint.exists():
                        continue
                    item = cache.get(cache_key("policy", checkpoint_signature(checkpoint), anchor_seed, future_seed, 252))
                    if item is not None:
                        paths[display].append(relative_wealth_path(item[0], baseline_returns))
        fig, axis = plt.subplots(figsize=(10, 6))
        x = np.arange(1, 253)
        for display, values in paths.items():
            array = np.stack(values)
            axis.plot(x, array.mean(axis=0), lw=2.2, color=COLORS[display], label=display)
            axis.fill_between(x, np.quantile(array, 0.10, axis=0), np.quantile(array, 0.90, axis=0), color=COLORS[display], alpha=0.12)
        axis.axhline(0, color="black", linestyle="--", label="Constrained-neutral baseline")
        axis.set_xlabel("Future market step"); axis.set_ylabel("Relative wealth")
        axis.set_title("Selected Cross-Version Holdout Paths\nMean with 10–90% branch band")
        _percent_axis(axis); axis.grid(True); axis.legend()
        fig.tight_layout(); fig.savefig(output_dir / "overview_03_selected_relative_wealth_paths.png"); plt.close(fig)
    finally:
        cache.close()


def plot_diagnostics(specs, summaries, output_dir: Path) -> None:
    factorial_ids = {"v34_standalone_m2_e3", "v34_standalone_m2_e4", "v34_standalone_m8_e3", "v34_standalone_m8_e4"}
    histories = defaultdict(list)
    for spec in specs:
        if spec.condition_id not in factorial_ids:
            continue
        history = load_validation_history(spec.run_dir)
        if history:
            histories[spec.condition].append(history)
    fig, axis = plt.subplots(figsize=(10, 6))
    for condition, seed_histories in histories.items():
        common_updates = sorted(set.intersection(*[{int(row["update"]) for row in history} for history in seed_histories]))
        if not common_updates:
            continue
        values = np.asarray([[next(float(row["validation_relative_wealth"]) for row in history if int(row["update"]) == update) for update in common_updates] for history in seed_histories])
        color = COLORS.get(condition, "#4e79a7")
        axis.plot(common_updates, values.mean(axis=0), color=color, lw=2, label=condition)
        axis.fill_between(common_updates, values.min(axis=0), values.max(axis=0), color=color, alpha=0.12)
    axis.axhline(0, color="#777777", lw=0.8)
    axis.set_xlabel("Training update"); axis.set_ylabel("Validation relative wealth")
    axis.set_title("V3.4 Validation History: Seed Mean and Range")
    _percent_axis(axis); axis.grid(True); axis.legend()
    fig.tight_layout(); fig.savefig(output_dir / "diagnostics_01_validation_history.png"); plt.close(fig)

    v34 = [row for row in summaries if row["version"] == "v3.4" and row["checkpoint_type"] == "best_return"]
    conditions = sorted({str(row["condition_id"]) for row in v34 if sum(r["condition_id"] == row["condition_id"] for r in v34) == 3})
    grouped = _seed_metric(v34, conditions, "best_return")
    fig, axis = plt.subplots(figsize=(11, 5.8))
    _dot_panel(axis, grouped, conditions, "mean_relative_wealth", "V3.4 Seed Variability", "Holdout relative wealth")
    fig.tight_layout(); fig.savefig(output_dir / "diagnostics_02_seed_variability.png"); plt.close(fig)

    valid = [row for row in summaries if row["validation_relative_wealth"] != "" and row["checkpoint_type"] == "best_return"]
    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    for version, marker in (("v2.6", "o"), ("v3.2", "s"), ("v3.3", "^"), ("v3.4", "D")):
        part = [row for row in valid if row["version"] == version]
        axis.scatter([float(row["validation_relative_wealth"]) for row in part], [float(row["mean_relative_wealth"]) for row in part], label=version, marker=marker, alpha=0.75)
    all_values = [float(row[field]) for row in valid for field in ("validation_relative_wealth", "mean_relative_wealth")]
    low, high = min(all_values), max(all_values)
    axis.plot([low, high], [low, high], "--", color="#777777")
    axis.set_xlabel("Selected validation relative wealth"); axis.set_ylabel("Independent holdout relative wealth")
    axis.set_title("Validation-to-Holdout Generalization")
    _percent_axis(axis, x=True); _percent_axis(axis); axis.grid(True); axis.legend()
    fig.tight_layout(); fig.savefig(output_dir / "diagnostics_03_validation_to_holdout_gap.png"); plt.close(fig)


def condition_summary(seed_rows: list[dict[str, object]]) -> dict[str, object]:
    output = {}
    keys = sorted({(str(row["version"]), str(row["condition_id"]), str(row["checkpoint_type"])) for row in seed_rows})
    for version, condition_id, checkpoint in keys:
        rows = [row for row in seed_rows if row["version"] == version and row["condition_id"] == condition_id and row["checkpoint_type"] == checkpoint]
        output[f"{version}:{condition_id}:{checkpoint}"] = {
            "condition": rows[0]["condition"], "seed_count": len(rows),
            "mean_relative_wealth_across_seeds": float(np.mean([float(row["mean_relative_wealth"]) for row in rows])),
            "min_relative_wealth_across_seeds": float(np.min([float(row["mean_relative_wealth"]) for row in rows])),
            "max_relative_wealth_across_seeds": float(np.max([float(row["mean_relative_wealth"]) for row in rows])),
            "mean_common_feasible_rate": float(np.mean([float(row["common_feasible_rate"]) for row in rows])),
            "mean_max_drawdown": float(np.mean([float(row["mean_max_drawdown"]) for row in rows])),
            "mean_cvar10_relative_wealth": float(np.mean([float(row["cvar10_relative_wealth"]) for row in rows])),
        }
    return output


def write_report(path: Path, summaries, effects, output_dir: Path) -> None:
    def aggregate(condition_id: str, checkpoint: str = "best_return"):
        rows = [row for row in summaries if row["condition_id"] == condition_id and row["checkpoint_type"] == checkpoint]
        if not rows:
            return None
        return {
            "return": float(np.mean([float(row["mean_relative_wealth"]) for row in rows])),
            "feasible": float(np.mean([float(row["common_feasible_rate"]) for row in rows])),
            "drawdown": float(np.mean([float(row["mean_max_drawdown"]) for row in rows])),
            "seeds": len(rows),
        }

    def effect_summary(name: str, checkpoint: str = "best_return"):
        rows = [
            row for row in effects
            if row["comparison"] == name and row["checkpoint_type"] == checkpoint
        ]
        if not rows:
            return None
        deltas = np.asarray([float(row["mean_relative_wealth_delta"]) for row in rows])
        return {
            "return": float(np.mean(deltas)),
            "positive": int(np.sum(deltas > 0.0)),
            "count": len(rows),
            "feasible": float(np.mean([
                float(row["common_feasible_rate_delta"]) for row in rows
            ])),
            "drawdown": float(np.mean([
                float(row["mean_max_drawdown_delta"]) for row in rows
            ])),
        }

    def checkpoint_average(checkpoint: str):
        main_ids = {
            "v34_standalone_m2_e3", "v34_standalone_m2_e4",
            "v34_standalone_m8_e3", "v34_standalone_m8_e4",
            "v34_cf_reward_m2_e3", "v34_cf_reward_m2_e4",
        }
        rows = [
            row for row in summaries
            if row["condition_id"] in main_ids and row["checkpoint_type"] == checkpoint
        ]
        return {
            "return": float(np.mean([float(row["mean_relative_wealth"]) for row in rows])),
            "feasible": float(np.mean([float(row["common_feasible_rate"]) for row in rows])),
            "drawdown": float(np.mean([float(row["mean_max_drawdown"]) for row in rows])),
            "tail": float(np.mean([float(row["cvar10_relative_wealth"]) for row in rows])),
        }

    validation_gaps = {}
    for version in ("v2.6", "v3.2", "v3.3", "v3.4"):
        rows = [
            row for row in summaries
            if row["version"] == version
            and row["checkpoint_type"] == "best_return"
            and row["validation_relative_wealth"] != ""
        ]
        if rows:
            validation_gaps[version] = float(np.mean([
                float(row["mean_relative_wealth"])
                - float(row["validation_relative_wealth"])
                for row in rows
            ]))

    market_e3 = effect_summary("market_2_to_8_e3")
    market_e4 = effect_summary("market_2_to_8_e4")
    epochs_m2 = effect_summary("epochs_3_to_4_m2")
    epochs_m8 = effect_summary("epochs_3_to_4_m8")
    cf_e3 = effect_summary("counterfactual_e3")
    cf_e4 = effect_summary("counterfactual_e4")
    margin_085 = effect_summary("margin_085_vs_090")
    margin_080 = effect_summary("margin_080_vs_090")
    best_average = checkpoint_average("best_return")
    robust_average = checkpoint_average("robust_feasible")
    v33_m2_5040 = aggregate("v33_gaussian_m2_s5040_e3")
    v33_m2_10080 = aggregate("v33_gaussian_m2_s10080_e3")
    v33_m2_20160 = aggregate("v33_gaussian_m2_s20160_e3")
    v33_m8_10080 = aggregate("v33_gaussian_m8_s10080_e3")
    v33_cf_dirichlet = aggregate("v33_cf_reward_dirichlet_m2_s10080_e4")
    window_results = {
        lookback: aggregate(f"v34_window_{lookback}") for lookback in (20, 30, 40, 60)
    }
    lines = [
        "# V3.3–V3.4 Evaluation Meeting Guide", "",
        "## Evaluation Protocol", "",
        "All policies were evaluated on five independent anchors with 20 future one-year branches per anchor. Relative wealth is measured against the constrained-neutral CAOSD baseline. Risk comparisons use a common 0.90 drawdown margin, 0.05 floor, 0.10 cost scale, and 0.05 alpha-budget ratio.", "",
        "Best-return and robust-feasible-selected checkpoints are reported separately. A robust-feasible-selected checkpoint is not assumed to remain feasible on independent holdout markets.", "",
        "## Main V3.4 Results", "",
    ]
    for condition_id in ("v34_standalone_m2_e3", "v34_standalone_m2_e4", "v34_standalone_m8_e3", "v34_standalone_m8_e4", "v34_cf_reward_m2_e3", "v34_cf_reward_m2_e4"):
        result = aggregate(condition_id)
        if result:
            condition = next(row["condition"] for row in summaries if row["condition_id"] == condition_id)
            lines.append(f"- **{condition}:** {result['return']:+.2%} mean relative wealth, {result['feasible']:.0%} common feasible branches, {result['drawdown']:.2%} mean maximum drawdown across {result['seeds']} seeds.")

    lines.extend(["", "## Data-Supported Conclusions", ""])
    if market_e3 and market_e4:
        lines.append(
            f"- **More training markets do not have a uniform main effect.** At 3 epochs, "
            f"moving from 2 to 8 markets changed holdout relative wealth by "
            f"{market_e3['return']:+.2%} and was positive for {market_e3['positive']}/3 seeds. "
            f"At 4 epochs, the effect was {market_e4['return']:+.2%} and positive for "
            f"{market_e4['positive']}/3 seeds."
        )
    if epochs_m2 and epochs_m8:
        lines.append(
            f"- **A fourth optimization epoch is condition-dependent rather than reliably better.** "
            f"The paired effect was {epochs_m2['return']:+.2%} with 2 markets "
            f"({epochs_m2['positive']}/3 seeds positive), but {epochs_m8['return']:+.2%} "
            f"with 8 markets ({epochs_m8['positive']}/3 positive)."
        )
    if cf_e3 and cf_e4:
        lines.append(
            f"- **Counterfactual reward shows a return-risk trade-off, not a general return gain.** "
            f"At 3 epochs the treatment-minus-control return effect was "
            f"{cf_e3['return']:+.2%}, with the treatment higher for "
            f"{cf_e3['positive']}/3 seeds, while common feasibility changed by "
            f"{cf_e3['feasible']:+.1%} and maximum drawdown by {cf_e3['drawdown']:+.2%}. "
            f"At 4 epochs its return effect was {cf_e4['return']:+.2%} "
            f"({cf_e4['positive']}/3 positive)."
        )
    lines.append(
        f"- **Checkpoint selection creates a material risk-return trade-off.** Across the six "
        f"replicated V3.4 conditions, best-return checkpoints averaged "
        f"{best_average['return']:+.2%} relative wealth and {best_average['feasible']:.1%} "
        f"common feasibility. Robust-feasible-selected checkpoints averaged "
        f"{robust_average['return']:+.2%} and {robust_average['feasible']:.1%}, with mean "
        f"maximum drawdown changing from {best_average['drawdown']:.2%} to "
        f"{robust_average['drawdown']:.2%}."
    )
    if margin_085 and margin_080:
        lines.append(
            f"- **Tightening the training margin did not improve risk under the common rule in "
            f"this single-seed sweep.** Relative to margin 0.90, margin 0.85 changed return by "
            f"{margin_085['return']:+.2%} and common feasibility by "
            f"{margin_085['feasible']:+.1%}; margin 0.80 changed them by "
            f"{margin_080['return']:+.2%} and {margin_080['feasible']:+.1%}."
        )
    if validation_gaps:
        gap_text = ", ".join(
            f"{version} {gap:+.2%}" for version, gap in validation_gaps.items()
        )
        lines.append(
            f"- **Validation selection remains optimistic.** Mean holdout minus selected-validation "
            f"relative wealth was {gap_text}. V3.4 has the largest average gap in this comparison."
        )

    lines.extend(["", "## V3.3 Exploratory Evidence", ""])
    if all((v33_m2_5040, v33_m2_10080, v33_m2_20160, v33_m8_10080)):
        lines.extend([
            f"- With two training markets, increasing each path from 5,040 to 10,080 steps "
            f"raised holdout relative wealth from {v33_m2_5040['return']:+.2%} to "
            f"{v33_m2_10080['return']:+.2%}; extending again to 20,160 steps reduced it to "
            f"{v33_m2_20160['return']:+.2%}. Longer paths therefore did not improve monotonically.",
            f"- The 8x10,080 condition traded some return ({v33_m8_10080['return']:+.2%}) "
            f"for the strongest V3.3 market-design risk result: "
            f"{v33_m8_10080['feasible']:.0%} common feasibility and "
            f"{v33_m8_10080['drawdown']:.2%} mean maximum drawdown.",
        ])
    if v33_cf_dirichlet:
        lines.append(
            f"- Counterfactual-reward Dirichlet produced the highest V3.3 method-ablation "
            f"best-return result ({v33_cf_dirichlet['return']:+.2%}), but only "
            f"{v33_cf_dirichlet['feasible']:.0%} common feasibility. This is single-seed "
            f"evidence and should not be ranked above the replicated V3.4 conditions."
        )

    lines.extend(["", "## Window And Margin Findings", ""])
    if all(window_results.values()):
        lines.append(
            f"- Window 60 achieved the highest exploratory return "
            f"({window_results[60]['return']:+.2%}) but also the highest maximum drawdown "
            f"({window_results[60]['drawdown']:.2%}) and lowest common feasibility "
            f"({window_results[60]['feasible']:.0%}). Window 20 was more balanced at "
            f"{window_results[20]['return']:+.2%} return, "
            f"{window_results[20]['drawdown']:.2%} drawdown, and "
            f"{window_results[20]['feasible']:.0%} feasibility."
        )
    lines.append(
        "- The margin sweep is not evidence that a stricter training budget makes the learned "
        "policy safer: both stricter settings reduced common-0.90 feasibility relative to the "
        "matched margin-0.90 seed. The own-rule curves are retained to show how the policy was "
        "trained, while the common-rule curves support the cross-margin conclusion."
    )
    lines.extend(["", "## Figure Index", ""])
    captions = [
        ("overview_01_cross_version_return_risk.png", "Cross-version return-risk evidence"),
        ("overview_02_checkpoint_tradeoff.png", "Checkpoint return-risk trade-off"),
        ("overview_03_selected_relative_wealth_paths.png", "Selected holdout wealth paths"),
        ("v33_01_market_design_return.png", "V3.3 market-count and train-length study"),
        ("v33_03_method_ablation.png", "V3.3 method ablation"),
        ("v34_01_factorial_best_return.png", "V3.4 replicated factorial, best-return checkpoints"),
        ("v34_02_factorial_robust_feasible.png", "V3.4 robust-feasible-selected checkpoints"),
        ("v34_03b_factorial_robust_feasible_risk.png", "V3.4 robust-feasible-selected risk"),
        ("v34_04_factorial_paired_effects.png", "Paired market-count and epoch effects"),
        ("v34_05_credit_return.png", "Standalone versus counterfactual reward"),
        ("v34_05b_credit_robust_feasible.png", "Credit comparison, robust-feasible-selected checkpoints"),
        ("v34_08_window_sweep.png", "Observation-window sweep"),
        ("v34_08b_window_robust_feasible.png", "Observation-window robust-feasible-selected sweep"),
        ("v34_09_margin_sweep.png", "Drawdown-margin sweep"),
        ("v34_09b_margin_robust_feasible.png", "Drawdown-margin robust-feasible-selected sweep"),
        ("diagnostics_03_validation_to_holdout_gap.png", "Validation-to-holdout gap"),
    ]
    for filename, caption in captions:
        lines.extend([f"### {caption}", "", f"![{caption}](../evaluation/v3.3_v3.4_professor_meeting/{filename})", ""])
    lines.extend([
        "## Interpretation Rules", "",
        "- V3.4 replicated comparisons use all three seeds and are the main evidence.",
        "- V3.3, observation-window, and drawdown-margin results are single-seed exploratory evidence.",
        "- Branches from the same anchor are not treated as independent training replications.",
        "- Allocation feasibility is guaranteed by CAOSD and independently checked during evaluation.",
        "- A higher own-rule feasible rate under a stricter margin is not sufficient evidence by itself; the common 0.90 rule is used for cross-margin comparison.", "",
        "## Discussion Questions", "",
        "1. Do market-count or epoch effects remain consistent across all three seeds?",
        "2. Does counterfactual reward improve tail performance as well as mean return?",
        "3. Is the robust-feasible-selected checkpoint giving a useful return-risk trade-off on unseen anchors?",
        "4. Are the window and margin trends strong enough to justify multi-seed confirmation?",
        "5. How much checkpoint-selection optimism remains between validation and independent holdout?", "",
        "## Limitations", "",
        "- V3.3 and the V3.4 window/margin sweeps use one seed. The larger paired holdout cannot replace independent training seeds.",
        "- Three V3.4 seeds support direction and variability checks, but not strong significance claims.",
        "- Robust-feasible-selected means selected by training validation criteria; average common holdout feasibility remains below 50%, so it is not a safety guarantee.",
        "- The evidence comes from the synthetic market and fixed allocation constraints. Real-market robustness and alternative constraint definitions remain open.",
        "- Window and margin trends should be treated as hypotheses for a replicated follow-up, not final hyperparameter choices.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_outputs(specs, rows, output_dir: Path, report_path: Path, anchor_seeds, branch_count) -> dict[str, object]:
    summaries = summarize_seeds(specs, rows)
    effects = paired_effects(rows)
    write_csv(output_dir / "seed_summary.csv", summaries)
    write_csv(output_dir / "paired_effects.csv", effects)
    _style()
    plot_v33(summaries, output_dir)
    plot_v34(summaries, effects, output_dir)
    plot_overview(specs, summaries, rows, output_dir)
    plot_diagnostics(specs, summaries, output_dir)
    summary = {
        "protocol": {
            "anchor_seeds": anchor_seeds, "branches_per_anchor": branch_count,
            "future_steps": 252, "baseline": "constrained-neutral CAOSD",
            "common_drawdown_rule": {"margin": 0.90, "floor": 0.05, "cost_scale": 0.10, "alpha_budget_ratio": 0.05},
        },
        "manifest_entries": len(specs),
        "unique_run_directories": len({str(spec.run_dir.resolve()) for spec in specs}),
        "branch_metric_rows": len(rows),
        "conditions": condition_summary(summaries),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_path, summaries, effects, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professor-meeting evaluation for V2.6–V3.4.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--anchor-seeds", type=int, nargs="+", default=ANCHOR_SEEDS_DEFAULT)
    parser.add_argument("--branches-per-anchor", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Use one anchor and one branch.")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate tables/plots from branch_metrics.csv.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    specs = discover_specs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_rows(specs)
    write_csv(args.output_dir / "evaluation_manifest.csv", manifest)
    anchor_seeds = [ANCHOR_SEEDS_DEFAULT[0]] if args.smoke else list(args.anchor_seeds)
    branch_count = 1 if args.smoke else int(args.branches_per_anchor)
    if args.plots_only:
        rows = read_csv(args.output_dir / "branch_metrics.csv")
    else:
        rows = evaluate(specs, args.output_dir, anchor_seeds=anchor_seeds, branch_count=branch_count)
    summary = generate_outputs(specs, rows, args.output_dir, args.report, anchor_seeds, branch_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
