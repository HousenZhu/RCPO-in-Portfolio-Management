"""Build the project report's three tables and Figure 1 from explicit inputs.

Run with --manifest pointing to a JSON file. See the usage text for its fields.
Paths in the manifest are resolved relative to the manifest file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.evaluation import load_checkpoint_for_evaluation
from rcpo_portfolio.market import generate_continuation_split, generate_market_split
from rcpo_portfolio.paper_evaluation import (
    BASELINE, CREDIT_METHODS, METHODS, STANDALONE, batched_rollout, paired_differences,
    plot_learning, rollout, score_trajectory, slice_market, summarize, write_csv,
)

MODES = {
    "caosd_rcpo_shared": "shared_global_branch",
    STANDALONE: "standalone_reward_global_cost",
    "caosd_rcpo_cf_r": "counterfactual_open_loop_reward_global_cost",
    "caosd_rcpo_cf_c": "standalone_reward_counterfactual_open_loop_cost",
    "caosd_rcpo_cf_rc": "counterfactual_open_loop_reward_cost",
}
TEST_CHECKPOINTS = {"checkpoint_best_return.pt", "checkpoint_best_feasible.pt", "checkpoint_best.pt"}
PROTOCOL_FIELDS = {
    "id", "surrogate_type", "outer_branch_weight", "entropy_weight", "critic_schema",
    "reward_scaling", "real_transition_budget", "training_code_revision",
}
RESULT_FIELDS = (
    "relative_wealth", "actual_cumulative_return", "terminal_wealth", "cost_gap",
    "path_feasible", "max_drawdown", "average_turnover", "allocation_violation_step_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Manifest (JSON):
  reference_config, actor_family (gaussian or dirichlet), protocol_id
  test and validation: {id, markets: [{id, anchor_seed, continuation_seed}]}
  runs: [{algorithm_id, run_dir, training_seed, test_checkpoint,
          validation_checkpoints: [{checkpoint, training_real_transition_count}]}]
  matched_runs: optional independent cohort with the same run fields for Table 2.
    Each checkpoint must carry matching training-side matched_credit_protocol metadata.
  compute: optional measured records keyed by algorithm_id and training_seed.
    Supply hardware_id, timing_protocol_id, training_real_transition_count,
    shadow_transition_count, optimizer_step_count, and measurements as repeats
    (training_wall_time_seconds and peak_training_memory_bytes). Missing stays blank.
  Optional risk: {margin, floor, cost_scale, alpha_budget_ratio}.
  Optional decoding, action_repeats, action_seed, cost_tolerance, allocation_tolerance,
  bootstrap_repeats, bootstrap_seed, timing_warmup, timing_repeats, anchor_steps,
  and evaluation_device (cpu by default, or cuda).
Validation checkpoints are supplied explicitly. Every point needs its actual
interaction count, including missing checkpoints. Provide a fresh output directory.
""",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def asset_groups(env: PortfolioEnv) -> list[tuple[list[int], float]]:
    preset = env.resolved_constraint_preset()
    return [
        (getattr(env.config, f"allocation_constraint_{i}_indices"),
         float(preset[f"allocation_constraint_{i}_min_weight"]))
        for i in (1, 2)
    ]


def market_signature(config) -> dict:
    return {
        k: v for k, v in asdict(config.market).items()
        if k not in {"lookback", "train_generation_lookback", "train_steps",
                     "train_market_count", "validation_steps", "test_steps"}
    }


def build_markets(reference, splits, lookback: int, horizon: int, anchor_steps: int):
    if anchor_steps < 1:
        raise ValueError("anchor_steps must be positive.")
    canonical = replace(reference.market, lookback=lookback, train_generation_lookback=lookback)
    anchors = {}
    seen_seeds = {}
    result = {}
    for split, spec in splits.items():
        branches = []
        seen_ids = set()
        for entry in spec["markets"]:
            market_id = entry["id"]
            anchor_seed = int(entry["anchor_seed"])
            continuation_seed = int(entry["continuation_seed"])
            if market_id in seen_ids or min(anchor_seed, continuation_seed) < 0:
                raise ValueError(f"Duplicate market id or invalid seed in {split}: {market_id}")
            seen_ids.add(market_id)
            key = (anchor_seed, continuation_seed)
            if key in seen_seeds or any(
                other_anchor == anchor_seed and other_split != split
                for (other_anchor, _), other_split in seen_seeds.items()
            ):
                raise ValueError("Test and validation need distinct anchors and unique branches.")
            seen_seeds[key] = split
            if anchor_seed not in anchors:
                anchors[anchor_seed] = generate_market_split(canonical, anchor_steps, anchor_seed)
            market = generate_continuation_split(
                canonical, anchors[anchor_seed], horizon, continuation_seed
            )
            branches.append((market_id, anchor_seed, continuation_seed, market))
        if not branches:
            raise ValueError(f"No markets specified for {split}.")
        result[split] = branches
    return result


def load_policy(entry, manifest_dir: Path, checkpoint: str, family: str, device: str, market, *, matched=False):
    method = entry["algorithm_id"]
    run = (manifest_dir / entry["run_dir"]).resolve()
    if Path(checkpoint).name != checkpoint or not (run / checkpoint).is_file():
        raise FileNotFoundError(run / checkpoint)
    config, metadata, model, _ = load_checkpoint_for_evaluation(
        run, checkpoint, evaluation_device=device, shared_market=market
    )
    if metadata["seed"] != int(entry["training_seed"]):
        raise ValueError(f"Training seed does not match checkpoint: {run / checkpoint}")
    actual_family = "dirichlet" if "dirichlet" in metadata["policy_architecture"] else "gaussian"
    soft = method == "rcpo_allocation_penalty"
    if config.environment.action_mode != ("softmax" if soft else "simplex_decomposition"):
        raise ValueError(f"Action mode does not match paper method {method}.")
    if metadata["algo"] != ("ppo_unconstrained" if method == "caosd_ppo" else "rcpo"):
        raise ValueError(f"Training algorithm does not match paper method {method}.")
    if soft and metadata["constraint_mode"] not in {"allocation_drawdown", "allocation_relative_drawdown"}:
        raise ValueError("The paper's soft RCPO baseline must train on allocation and drawdown cost.")
    if not soft and actual_family != family:
        raise ValueError(f"{method} actor family {actual_family} does not match {family}.")
    if method in MODES and metadata["branch_credit_mode"] != MODES[method]:
        raise ValueError(f"Branch credit does not match paper method {method}.")
    if method.startswith("caosd_rcpo_cf_") and metadata["counterfactual_semantics_version"] != 2:
        raise ValueError(f"{method} requires a single-step CF checkpoint.")
    protocol = metadata["matched_credit_protocol"]
    surrogate = (
        protocol["surrogate_type"]
        if isinstance(protocol, dict) and PROTOCOL_FIELDS <= protocol.keys()
        else "joint" if metadata["branch_credit_mode"] == "global" else "conditional_branch"
    )
    if matched and (not isinstance(protocol, dict) or not PROTOCOL_FIELDS <= protocol.keys()
                    or surrogate != "conditional_branch"):
        raise ValueError("Checkpoint lacks a verified conditional-branch matched protocol.")
    return run, config, metadata, model, actual_family, surrogate


def evaluation_env(config, market, horizon):
    return PortfolioEnv(
        replace(config.environment, episode_length=horizon),
        slice_market(market, config.market.lookback, horizon),
        config.market,
        collect_branch_credit=False,
    )


def policy_function(model, env, deterministic):
    if model is None:
        action = env.constrained_neutral_action()
        return lambda _obs: action
    device = next(model.parameters()).device

    @torch.inference_mode()
    def decide(obs):
        inputs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return model.get_policy_output(inputs, deterministic=deterministic).action[0].cpu().numpy()

    return decide


def decision_samples(model, env, deterministic, warmup, repeats):
    obs, _ = env.reset(options={"start_index": env.lookback})
    decide = policy_function(model, env, deterministic)
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    samples = []
    for i in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        env._weights_from_action(decide(obs))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= warmup:
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def checkpoint_transitions(config, metadata) -> int:
    optimization = config.ppo if metadata["algo"] == "ppo_unconstrained" else config.optimization
    return metadata["completed_updates"] * optimization.rollout_steps


def validate_runs(entries, methods):
    indices = defaultdict(set)
    for entry in entries:
        method = entry["algorithm_id"]
        seed = int(entry["training_seed"])
        if seed < 0 or method not in methods or seed in indices[method]:
            raise ValueError(f"Invalid or duplicate method and seed: {method}, {seed}")
        if entry["test_checkpoint"] not in TEST_CHECKPOINTS:
            raise ValueError("Test checkpoint must be selected by validation before evaluation.")
        indices[method].add(seed)
    if set(indices) != set(methods) or len({frozenset(s) for s in indices.values()}) != 1:
        raise ValueError("Specify all required methods with the same training seed set.")


def revision():
    files = (
        Path(__file__).resolve(),
        ROOT / "src/rcpo_portfolio/paper_evaluation.py",
        ROOT / "src/rcpo_portfolio/evaluation.py",
        ROOT / "src/rcpo_portfolio/env.py",
    )
    source_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"commit": commit, "working_tree_dirty": dirty, "source_sha256": source_hashes}


def run_suite(manifest, manifest_dir, output, *, v41=False):
    family = manifest["actor_family"]
    if family not in {"gaussian", "dirichlet"}:
        raise ValueError("Select one simplex actor family per evaluation.")
    decode = manifest.get("decoding", "deterministic")
    action_repeats = int(manifest.get("action_repeats", 1))
    action_seed = int(manifest.get("action_seed", 20261122))
    bootstrap = int(manifest.get("bootstrap_repeats", 5000))
    bootstrap_seed = int(manifest.get("bootstrap_seed", 3628))
    timing_warmup = int(manifest.get("timing_warmup", 20))
    timing_repeats = int(manifest.get("timing_repeats", 100))
    evaluation_device = manifest.get("evaluation_device", "cpu")
    if evaluation_device not in {"cpu", "cuda"}:
        raise ValueError("Use a single explicit evaluation device for all checkpoints.")
    cost_tolerance = float(manifest.get("cost_tolerance", 1e-10))
    allocation_tolerance = float(manifest.get("allocation_tolerance", 1e-6))
    if decode not in {"deterministic", "stochastic"} or (decode == "deterministic" and action_repeats != 1):
        raise ValueError("Deterministic decoding takes exactly one action repeat.")
    if min(action_repeats, bootstrap, timing_repeats) < 1 or timing_warmup < 0:
        raise ValueError("Invalid repeat or warmup count.")
    if not all(np.isfinite(x) and x >= 0 for x in (cost_tolerance, allocation_tolerance)):
        raise ValueError("Tolerances must be finite and nonnegative.")
    reference = load_config(manifest_dir / manifest["reference_config"])
    sync_rcpo_constraint_settings(reference)
    risk = {
        "margin": reference.environment.benchmark_drawdown_margin,
        "floor": reference.environment.drawdown_budget_floor,
        "cost_scale": reference.environment.drawdown_cost_scale,
        "alpha_budget_ratio": reference.rcpo.alpha_budget_ratio,
        **manifest.get("risk", {}),
    }
    if set(risk) != {"margin", "floor", "cost_scale", "alpha_budget_ratio"} or not all(
        np.isfinite(value) and value >= 0 for value in risk.values()
    ) or min(risk["margin"], risk["cost_scale"]) <= 0:
        raise ValueError("Invalid common drawdown rule.")
    runs = manifest["runs"]
    validate_runs(runs, set(METHODS) - {BASELINE})
    if any(not entry["validation_checkpoints"] for entry in runs):
        raise ValueError("List the intended Figure 1 checkpoints for each run, including missing files.")
    matched = manifest.get("matched_runs", [])
    if matched:
        validate_runs(matched, set(CREDIT_METHODS))
    max_lookback = max(
        reference.market.lookback,
        *(load_config(manifest_dir / entry["run_dir"] / "config_snapshot.yaml").market.lookback
          for entry in [*runs, *matched]),
    )
    horizon = reference.environment.episode_length
    if horizon < 1:
        raise ValueError("Evaluation horizon must be positive.")
    splits = {name: manifest[name] for name in ("test", "validation")}
    if splits["test"]["id"] == splits["validation"]["id"]:
        raise ValueError("Test and validation market sets must use distinct ids.")
    markets = build_markets(reference, splits, max_lookback, horizon,
                            int(manifest.get("anchor_steps", reference.evaluation.fixed_anchor_steps)))
    first_market = markets["test"][0][-1]
    reference_env = evaluation_env(reference, first_market, horizon)
    groups = asset_groups(reference_env)
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    rows = []
    baseline = {}
    figure_raw = []
    training_sources = []
    latency = defaultdict(list)
    selected_latency = defaultdict(list)
    compute_counts = defaultdict(list)
    test_cache = {}
    alternative_results = []

    def record(path, baseline_path, **identity):
        summary, steps = score_trajectory(
            path, baseline_path, risk=risk, groups=groups,
            cost_tolerance=cost_tolerance, allocation_tolerance=allocation_tolerance,
        )
        trace = f"trajectories/{len(rows):08d}.npz"
        np.savez_compressed(output / trace, **steps)
        row = {**identity, **summary, "trajectory": trace}
        row["allocation_violation_step_rate"] = summary["allocation_violation_step_count"] / horizon
        if identity["algorithm_id"] != "rcpo_allocation_penalty" and summary["allocation_violation_step_count"]:
            raise ValueError(f"Hard allocation violated: {identity}")
        rows.append(row)
        return row

    for split, branches in markets.items():
        for market_id, anchor, _, market in branches:
            env = evaluation_env(reference, market, horizon)
            path = rollout(env, policy_function(None, env, True), horizon)
            baseline[split, market_id] = path
            record(path, path, cohort="main", algorithm_id=BASELINE, actor_family="static",
                   protocol_id=manifest["protocol_id"], surrogate_type="static",
                   training_seed=None, checkpoint_id=None, split=split, market_id=market_id,
                   market_set_id=splits[split]["id"], anchor_seed=anchor, action_seed=None)
    latency[BASELINE, "cpu"].extend(
        decision_samples(None, reference_env, True, timing_warmup, timing_repeats)
    )

    def evaluate_checkpoint(entry, checkpoint, split, cohort):
        method = entry["algorithm_id"]
        run_dir = (manifest_dir / entry["run_dir"]).resolve()
        cache_key = (method, run_dir, checkpoint, int(entry["training_seed"]))
        if split == "test" and cohort == "matched" and cache_key in test_cache:
            main_rows, count, metadata = test_cache[cache_key]
            protocol = metadata["matched_credit_protocol"]
            if (not isinstance(protocol, dict) or not PROTOCOL_FIELDS <= protocol.keys()
                    or protocol["surrogate_type"] != "conditional_branch"):
                raise ValueError("Checkpoint lacks a verified conditional-branch matched protocol.")
            matched_rows = [{**row, "cohort": cohort} for row in main_rows]
            rows.extend(matched_rows)
            return matched_rows, count, metadata
        run, config, metadata, model, actual_family, surrogate = load_policy(
            entry, manifest_dir, checkpoint, family, evaluation_device, first_market,
            matched=cohort == "matched",
        )
        env = evaluation_env(config, first_market, horizon)
        if (market_signature(config) != market_signature(reference)
                or asset_groups(env) != groups
                or config.environment.transaction_cost_bps != reference.environment.transaction_cost_bps
                or config.environment.episode_length != horizon
                or not np.allclose(env.initial_portfolio_weights(), reference_env.initial_portfolio_weights(), atol=1e-7, rtol=0)):
            raise ValueError(f"Incompatible portfolio task or trading protocol: {run}")
        optimization = config.ppo if metadata["algo"] == "ppo_unconstrained" else config.optimization
        completed_transitions = checkpoint_transitions(config, metadata)
        source = {
            "cohort": cohort, "algorithm_id": method, "training_seed": metadata["seed"],
            "split": split, "run_dir": str(run), "checkpoint": checkpoint,
            "checkpoint_sha256": hashlib.sha256((run / checkpoint).read_bytes()).hexdigest(),
            "credit_mode": metadata["branch_credit_mode"], "actor_family": actual_family,
            "surrogate_type": surrogate, "constraint_mode": metadata["constraint_mode"],
            "constraint_semantics": metadata["constraint_semantics"],
            "alpha": metadata["alpha"], "alpha_budget_ratio": metadata["alpha_budget_ratio"],
            "lambda_value": metadata["lambda_value"],
            "allocation_cost_scale": metadata["allocation_constraint_cost_scale"],
            "combined_drawdown_cost_weight": metadata["combined_drawdown_cost_weight"],
            "training_drawdown_cost_scale": metadata["drawdown_cost_scale"],
            "training_budget_floor": config.environment.drawdown_budget_floor,
            "training_benchmark_margin": config.environment.benchmark_drawdown_margin,
            "training_benchmark_mode": config.environment.drawdown_benchmark_mode,
            "training_lambda_gap_mode": config.rcpo.lambda_gap_mode,
            "counterfactual_semantics_version": metadata["counterfactual_semantics_version"],
            "matched_credit_protocol": metadata["matched_credit_protocol"],
        }
        if method == "rcpo_allocation_penalty":
            source.update(
                training_cost_formula="allocation_cost + combined_drawdown_cost_weight * drawdown_cost",
                allocation_cost_formula="sum(squared shortfall divided by squared group minimum) / allocation_cost_scale",
                allocation_effective_multiplier=metadata["lambda_value"],
                drawdown_effective_multiplier=metadata["lambda_value"] * metadata["combined_drawdown_cost_weight"],
                cost_budget_scope="one shared budget for the combined cost",
            )
        training_sources.append(source)
        if split == "test":
            summary_path = run / "training_summary.json"
            updates = json.loads(summary_path.read_text(encoding="utf-8")).get("completed_updates") if summary_path.exists() else None
            full_transitions = None if updates is None else int(updates) * optimization.rollout_steps
            metadata["full_training_transitions"] = full_transitions
            if cohort == "main" or v41:
                samples = decision_samples(model, env, decode == "deterministic", timing_warmup, timing_repeats)
                if v41:
                    selected_latency[cohort, metadata["seed"], method, metadata["device"]].extend(samples)
                else:
                    latency[method, metadata["device"]].extend(samples)
            if cohort == "main":
                compute_counts[method].append(full_transitions)
        def record_episode(path, market_id, anchor, sample_seed):
            return record(
                    path, baseline[split, market_id], cohort=cohort, algorithm_id=method,
                    actor_family=actual_family, protocol_id=manifest["protocol_id"],
                    surrogate_type=surrogate, credit_mode=metadata["branch_credit_mode"],
                    training_seed=metadata["seed"],
                    checkpoint_id=checkpoint, split=split, market_id=market_id,
                    market_set_id=splits[split]["id"], anchor_seed=anchor, action_seed=sample_seed,
            )

        branches = markets[split]
        environments = []
        for _, _, _, market in branches:
            env = evaluation_env(config, market, horizon)
            np.testing.assert_array_equal(env.market.risky_returns[-horizon:], market.risky_returns[-horizon:])
            environments.append(env)
        if v41:
            paths = batched_rollout(environments, model, horizon)
            results = [
                record_episode(path, branch[0], branch[1], None)
                for path, branch in zip(paths, branches, strict=True)
            ]
        else:
            results = []
            device = next(model.parameters()).device
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            for (market_id, anchor, continuation, _), env in zip(branches, environments, strict=True):
                decide = policy_function(model, env, decode == "deterministic")
                for repeat in range(action_repeats):
                    sample_seed = None
                    if decode == "stochastic":
                        sample_seed = int(np.random.SeedSequence([action_seed, anchor, continuation, repeat]).generate_state(1)[0])
                    with torch.random.fork_rng(devices=devices):
                        if sample_seed is not None:
                            torch.random.default_generator.manual_seed(sample_seed)
                            if devices:
                                with torch.cuda.device(devices[0]):
                                    torch.cuda.manual_seed(sample_seed)
                        path = rollout(env, decide, horizon)
                    results.append(record_episode(path, market_id, anchor, sample_seed))
        metadata["training_settings"] = {
            "optimization": asdict(optimization),
            "market": asdict(config.market),
            "network": {
                k: v for k, v in asdict(config.network).items()
                if k != "branch_credit_mode" and not k.startswith("counterfactual_")
            },
            "environment": {
                k: v for k, v in asdict(config.environment).items()
                if k not in {
                    "standalone_branch_reward_only", "counterfactual_branch_credit_enabled",
                    "standalone_branch_cost_enabled",
                }
            },
            "rcpo": asdict(config.rcpo),
            "reward_noise": asdict(config.reward_noise),
            "reward_correction": asdict(config.reward_correction),
            "checkpoint_selection": asdict(config.evaluation),
        }
        if split == "test" and cohort == "main":
            test_cache[cache_key] = (results, completed_transitions, metadata)
        return results, completed_transitions, metadata

    for index, entry in enumerate(runs, start=1):
        if v41:
            print(f"[v4.1] {index}/{len(runs)} {entry['algorithm_id']} seed {entry['training_seed']}", flush=True)
        evaluate_checkpoint(entry, entry["test_checkpoint"], "test", "main")
        if v41:
            alternative_results.extend(
                evaluate_checkpoint(entry, "checkpoint_best_feasible.pt", "test", "best_feasible")[0]
            )
        seen_counts = set()
        for point in entry["validation_checkpoints"]:
            checkpoint = point["checkpoint"]
            count = int(point["training_real_transition_count"])
            if Path(checkpoint).name != checkpoint or count < 0 or count in seen_counts:
                raise ValueError("Each validation run needs unique, nonnegative transition counts.")
            seen_counts.add(count)
            figure_point = {
                "algorithm_id": entry["algorithm_id"], "actor_family": family,
                "training_seed": int(entry["training_seed"]), "checkpoint_id": checkpoint,
                "training_real_transition_count": count,
                "validation_market_set_id": splits["validation"]["id"],
            }
            if (manifest_dir / entry["run_dir"] / checkpoint).is_file():
                values, actual_count, metadata = evaluate_checkpoint(entry, checkpoint, "validation", "main")
                if count != actual_count:
                    raise ValueError(f"Validation transition count disagrees with checkpoint: {entry['run_dir']}/{checkpoint}")
                figure_point.update(
                    actor_family="dirichlet" if "dirichlet" in metadata["policy_architecture"] else "gaussian",
                    mean_relative_wealth=float(np.mean([v["relative_wealth"] for v in values])),
                    mean_cost_gap=float(np.mean([v["cost_gap"] for v in values])),
                    status="available",
                )
            else:
                values = []
                figure_point.update(status="missing", mean_relative_wealth=None, mean_cost_gap=None)
            figure_raw.append((figure_point, values))

    matched_results = defaultdict(list)
    matched_protocol = None
    if matched:
        contracts = defaultdict(list)
        for entry in matched:
            values, count, metadata = evaluate_checkpoint(
                entry, entry["test_checkpoint"], "test", "matched"
            )
            contract = metadata["matched_credit_protocol"]
            if matched_protocol is None:
                matched_protocol = contract
            budget = int(contract["real_transition_budget"])
            if (contract != matched_protocol or count > budget
                    or metadata["full_training_transitions"] != budget):
                raise ValueError("Table 2 checkpoints have different protocols or exceed the declared training budget.")
            contracts[metadata["seed"]].append(metadata["training_settings"])
            matched_results[entry["algorithm_id"]].extend(values)
        for settings in contracts.values():
            if any(other != settings[0] for other in settings[1:]):
                raise ValueError("Table 2 training settings differ within a matched seed.")

    if v41:
        from v41_outputs import write_v41_outputs

        write_v41_outputs(
            output, rows, alternative_results, figure_raw, matched_results, matched_protocol,
            family, {**selected_latency, (None, None, BASELINE, "cpu"): latency[BASELINE, "cpu"]},
            compute_counts, manifest.get("compute", []), bootstrap, bootstrap_seed,
        )
    else:
        write_results(
            output, rows, figure_raw, matched_results, matched_protocol, family, latency,
            compute_counts, manifest.get("compute", []), bootstrap, bootstrap_seed,
        )
    (output / "evaluation_metadata.json").write_text(json.dumps({
        "illustrative": False, "manifest": manifest, "comparison_actor_family": family,
        "policy_decoding": decode, "evaluation_device": evaluation_device, "code_revision": revision(),
        "wealth_source": "environment_recorded_growth", "wealth_growth_floor": 1e-12,
        "relative_wealth_baseline": BASELINE, "common_risk_rule": risk,
        "allocation_groups": groups, "cost_tolerance": cost_tolerance,
        "allocation_tolerance": allocation_tolerance, "action_repeats": action_repeats,
        "confidence_level": 0.95, "bootstrap_repeats": bootstrap, "bootstrap_seed": bootstrap_seed,
        "confidence_method": "Percentile bootstrap over training seeds and paired anchor blocks after averaging action repeats; null when no repeat dimension exists.",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                     "torch_version": torch.__version__, "torch_threads": torch.get_num_threads()},
        "decision_timing": {"batch_size": 1, "warmup": timing_warmup, "repeats_per_checkpoint": timing_repeats,
                            "boundary": "Policy API including critic heads, host transfer, and feasible action mapping."},
        "checkpoint_sources": training_sources,
    }, indent=2, allow_nan=False), encoding="utf-8")


def write_results(
    output, rows, figure_raw, matched_results, matched_protocol, family, latency,
    compute_counts, compute_records, bootstrap, bootstrap_seed,
):
    tables = []
    for method in METHODS:
        records = [
            r for r in rows
            if r["cohort"] == "main" and r["split"] == "test" and r["algorithm_id"] == method
        ]
        sources = {r["surrogate_type"] for r in records}
        if len(sources) != 1:
            raise ValueError(f"Mixed surrogate types for {method}.")
        tables.append({
            "algorithm_id": method, "display_name": METHODS[method],
            "actor_family": records[0]["actor_family"],
            "comparison_actor_family": family,
            "credit_mode": records[0].get("credit_mode"),
            "surrogate_type": sources.pop(),
            "training_seed_count": len({r["training_seed"] for r in records} - {None}),
            "market_count": len({r["market_id"] for r in records}),
            "action_repeats": len({r["action_seed"] for r in records if r["market_id"] == records[0]["market_id"]}),
            **summarize(records, RESULT_FIELDS, bootstrap, bootstrap_seed),
            "max_allocation_violation": max(r["max_allocation_violation"] for r in records),
        })
        tables[-1]["path_feasible_rate"] = tables[-1].pop("mean_path_feasible")
        tables[-1]["allocation_violation_step_rate"] = tables[-1].pop("mean_allocation_violation_step_rate")
    write_csv(output / "table_1_main_results.csv", tables)
    write_csv(output / "evaluation_episodes.csv", rows)

    paired_table = []
    paired_rows = []
    for method in CREDIT_METHODS:
        pairs = paired_differences(matched_results[method], matched_results[STANDALONE]) if matched_results else []
        paired_rows.extend(pairs)
        fields = ("delta_relative_wealth", "delta_cost_gap")
        estimates = summarize(pairs, fields, bootstrap, bootstrap_seed)
        paired_table.append({
            "comparison_id": f"{method}_vs_{STANDALONE}",
            "algorithm_id": method, "reference_algorithm_id": STANDALONE,
            "actor_family": family, "matched_protocol_id": matched_protocol["id"] if matched_protocol else None,
            "status": "reference" if pairs and method == STANDALONE else (
                "available" if pairs else "matched_checkpoints_not_supplied"
            ),
            "paired_training_seed_count": len({row["training_seed"] for row in pairs}),
            "paired_market_count": len({row["market_id"] for row in pairs}),
            **{f"{prefix}_{field}": estimates.get(f"{prefix}_{field}")
               for field in fields for prefix in ("mean", "ci_low", "ci_high")},
            **{f"{prefix}_delta_relative_wealth_pp": (
                100 * estimates[f"{prefix}_delta_relative_wealth"]
                if estimates.get(f"{prefix}_delta_relative_wealth") is not None else None
            ) for prefix in ("mean", "ci_low", "ci_high")},
        })
    write_csv(output / "table_2_matched_credit.csv", paired_table)
    if paired_rows:
        write_csv(output / "table_2_paired_episodes.csv", paired_rows)

    write_compute(output, family, latency, compute_counts, compute_records,
                  {row["algorithm_id"]: row["actor_family"] for row in tables})
    write_csv(output / "figure_1_checkpoint_metrics.csv", [point for point, _ in figure_raw])
    by_point = defaultdict(list)
    for point, records in figure_raw:
        key = (point["algorithm_id"], point["training_real_transition_count"], point["validation_market_set_id"])
        by_point[key].extend(records)
    figure = []
    for (method, count, market_set_id), records in sorted(by_point.items()):
        figure.append({
            "algorithm_id": method, "actor_family": next(row["actor_family"] for row in tables if row["algorithm_id"] == method),
            "training_real_transition_count": count, "validation_market_set_id": market_set_id,
            "effective_seed_count": len({r["training_seed"] for r in records}),
            **{
                f"{prefix}_{field}": None
                for field in ("relative_wealth", "cost_gap")
                for prefix in ("mean", "ci_low", "ci_high")
            },
            **summarize(records, ("relative_wealth", "cost_gap"), bootstrap, bootstrap_seed),
        })
    write_csv(output / "figure_1_checkpoint_summary.csv", figure)
    baseline = [r for r in rows if r["algorithm_id"] == BASELINE and r["split"] == "validation"]
    baseline_stats = summarize(baseline, ("relative_wealth", "cost_gap"), bootstrap, bootstrap_seed)
    write_csv(output / "figure_1_baseline.csv", [{
        "algorithm_id": BASELINE, "validation_market_set_id": baseline[0]["market_set_id"],
        "market_count": len(baseline), **baseline_stats,
    }])
    plot_learning(figure, baseline_stats, output)


def write_compute(output, family, latency, compute_counts, measured, actor_families):
    """Aggregate supplied training measurements and measured decision latencies."""
    measured_by_method = defaultdict(list)
    seen = set()
    for item in measured:
        method = item["algorithm_id"]
        seed = int(item["training_seed"])
        if method not in METHODS or method == BASELINE or (method, seed) in seen:
            raise ValueError(f"Invalid or duplicate training measurement: {method}, {seed}")
        seen.add((method, seed))
        if not item["hardware_id"] or not item["timing_protocol_id"]:
            raise ValueError("Compute records require hardware and timing protocol identifiers.")
        for name in ("training_real_transition_count", "shadow_transition_count", "optimizer_step_count",
                     "one_step_replacement_count"):
            count = item.get(name)
            if count is not None and (count < 0 or int(count) != count):
                raise ValueError(f"Invalid count for {method}: {name}")
        for field in ("training_wall_time_seconds", "peak_training_memory_bytes"):
            if any(not np.isfinite(x) or x <= 0 for x in (item.get(field) or [])):
                raise ValueError(f"Invalid positive timing or memory measurement for {method}: {field}")
        measured_by_method[method].append(item)

    def measured_stats(items, name):
        values = [value for item in items for value in (item.get(name) or [])]
        return (
            float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None,
            len(values),
        )

    def mean_count(values):
        return float(np.mean(values)) if values and all(x is not None for x in values) else None

    compute = []
    for method in METHODS:
        items = measured_by_method[method]
        hardware_protocol = {(item["hardware_id"], item["timing_protocol_id"]) for item in items}
        comparable_items = items if len(hardware_protocol) == 1 else []
        hardware, protocol = next(iter(hardware_protocol)) if comparable_items else (None, None)
        time_mean, time_std, time_count = measured_stats(comparable_items, "training_wall_time_seconds")
        memory_mean, memory_std, memory_count = measured_stats(comparable_items, "peak_training_memory_bytes")
        devices = {device: samples for (algo, device), samples in latency.items() if algo == method}
        if len(devices) != 1:
            raise ValueError(f"Use one evaluation device per algorithm: {method}.")
        device, samples = next(iter(devices.items()))
        real_counts = [item.get("training_real_transition_count") for item in items]
        actual_transitions = mean_count(real_counts if items else compute_counts[method])
        compute.append({
            "algorithm_id": method, "actor_family": actor_families[method],
            "comparison_actor_family": family, "hardware_id": hardware,
            "timing_protocol_id": protocol, "batch_size": 1, "inference_device": device,
            "training_measurement_status": (
                "mixed_hardware_or_protocol" if items and not comparable_items
                else "available" if comparable_items else "missing"
            ),
            "training_real_transition_count": 0 if method == BASELINE else actual_transitions,
            "training_transition_count_source": (
                "static_policy" if method == BASELINE else
                "measured_record" if items else "retained_updates_times_rollout_steps"
            ),
            "training_transition_counts_comparable": bool(items and actual_transitions is not None
                                                        and len(set(real_counts)) == 1),
            "shadow_transition_count": 0 if method == BASELINE else mean_count([item.get("shadow_transition_count") for item in items]),
            "one_step_replacement_count": 0 if method == BASELINE else mean_count([item.get("one_step_replacement_count") for item in items]),
            "optimizer_step_count": 0 if method == BASELINE else mean_count([item.get("optimizer_step_count") for item in items]),
            "training_wall_time_seconds": time_mean, "training_time_std_seconds": time_std,
            "training_time_sample_count": time_count,
            "peak_training_memory_bytes": memory_mean, "peak_training_memory_std_bytes": memory_std,
            "training_memory_sample_count": memory_count,
            "decision_latency_ms": float(np.mean(samples)),
            "decision_latency_std_ms": float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
            "decision_latency_sample_count": len(samples),
        })
    reference = next(row for row in compute if row["algorithm_id"] == "caosd_rcpo_shared")
    for row in compute:
        same_training_protocol = (
            row["hardware_id"] is not None and row["timing_protocol_id"] is not None
            and row["training_transition_counts_comparable"] and reference["training_transition_counts_comparable"]
            and all(row[key] == reference[key] for key in
                    ("hardware_id", "timing_protocol_id", "training_real_transition_count"))
        )
        for target, raw in (
            ("relative_training_time", "training_wall_time_seconds"),
            ("relative_peak_memory", "peak_training_memory_bytes"),
            ("relative_decision_latency", "decision_latency_ms"),
        ):
            comparable = (row["inference_device"] == reference["inference_device"]
                          if raw == "decision_latency_ms" else same_training_protocol)
            row[target] = row[raw] / reference[raw] if comparable and row[raw] is not None and reference[raw] else None
    write_csv(output / "table_3_compute.csv", compute)
    (output / "compute_measurements.json").write_text(json.dumps({
        "training_measurements": measured,
        "decision_measurements_ms": [
            {"algorithm_id": method, "device": device, "samples": samples}
            for (method, device), samples in latency.items()
        ],
    }, indent=2, allow_nan=False), encoding="utf-8")


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_suite(manifest, manifest_path.parent, args.output_dir.resolve())


if __name__ == "__main__":
    main()
