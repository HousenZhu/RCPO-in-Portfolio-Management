from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.devices import resolve_device
from rcpo_portfolio.profiling import TrainingProfiler
from rcpo_portfolio.trainer import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile simplex PPO/RCPO training time by training-loop section."
    )
    parser.add_argument(
        "--algo",
        choices=["all", "ppo_unconstrained", "rcpo"],
        default="all",
        help="Algorithm scenario to profile. Defaults to all PPO/RCPO simplex scenarios.",
    )
    parser.add_argument(
        "--constraint-drawdown",
        action="store_true",
        help="Accepted for parity with train.py; RCPO profiling always uses drawdown.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config YAML to use as the profiling base.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional profile output directory. Defaults to a temp folder.",
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    parser.add_argument(
        "--policy-architecture",
        choices=[
            "flat_gaussian",
            "simplex_branch_gaussian",
            "simplex_autoregressive_gaussian",
        ],
        default=None,
        help="Run one architecture. Omit to run both simplex branch architectures.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["softmax", "simplex_decomposition"],
        default="simplex_decomposition",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation/checkpoint scoring to isolate rollout and optimization speed.",
    )
    parser.add_argument(
        "--no-write-plots",
        action="store_true",
        help="Skip plot/final-artifact generation while still writing profile CSV/JSON.",
    )
    return parser.parse_args()


def _scenario_architectures(args: argparse.Namespace) -> list[str]:
    if args.policy_architecture is not None:
        return [args.policy_architecture]
    if args.action_mode == "softmax":
        return ["flat_gaussian"]
    if args.algo != "all":
        return ["simplex_branch_gaussian"]
    return ["simplex_branch_gaussian", "simplex_autoregressive_gaussian"]


def _scenario_algorithms(args: argparse.Namespace) -> list[str]:
    if args.algo == "all":
        return ["ppo_unconstrained", "rcpo"]
    return [args.algo]


def _configure_scenario(
    base_config,
    *,
    args: argparse.Namespace,
    algo: str,
    policy_architecture: str,
    scenario_output_root: Path,
) -> Any:
    config = copy.deepcopy(base_config)
    config.experiment.output_root = str(scenario_output_root)
    config.experiment.run_name = f"profile_{algo}_{policy_architecture}"
    config.experiment.seeds = [0]
    config.runtime.device = args.device
    config.environment.action_mode = args.action_mode
    config.network.policy_architecture = policy_architecture
    if algo == "rcpo":
        config.rcpo.constraint_mode = "max_drawdown"

    for optimization in [config.optimization, config.ppo]:
        optimization.total_updates = int(args.updates)
        optimization.rollout_steps = int(args.rollout_steps)
        optimization.epochs = int(args.epochs)
        optimization.minibatch_size = int(args.minibatch_size)
        optimization.early_stop_patience = None
        optimization.target_kl = None

    sync_rcpo_constraint_settings(config)
    return config


def _write_combined_summary(
    output_root: Path,
    scenario_summaries: list[dict[str, Any]],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenarios": scenario_summaries,
    }
    with (output_root / "profile_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.updates < 1:
        raise ValueError("--updates must be at least 1.")
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.minibatch_size < 1:
        raise ValueError("--minibatch-size must be at least 1.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else PROJECT_ROOT / "profile_outputs" / timestamp
    )
    output_root.mkdir(parents=True, exist_ok=True)
    training_output_root = output_root / "training_runs"
    base_config = load_config(args.config)
    scenario_summaries: list[dict[str, Any]] = []

    for algo in _scenario_algorithms(args):
        for policy_architecture in _scenario_architectures(args):
            scenario_name = f"{algo}_{policy_architecture}"
            print(f"\n[profile] scenario={scenario_name}", flush=True)
            scenario_output_root = training_output_root / scenario_name
            config = _configure_scenario(
                base_config,
                args=args,
                algo=algo,
                policy_architecture=policy_architecture,
                scenario_output_root=scenario_output_root,
            )
            device = resolve_device(config.runtime.device)
            profiler = TrainingProfiler(enabled=True, device=device)
            run_dirs = run_experiment(
                config=config,
                algo=algo,
                output_root=scenario_output_root,
                profiler=profiler,
                disable_artifacts=args.no_write_plots,
                skip_validation=args.skip_validation,
            )
            timing_json = output_root / f"{scenario_name}_timing.json"
            timing_csv = output_root / f"{scenario_name}_timing.csv"
            profiler.write_json(timing_json)
            profiler.write_csv(timing_csv)
            print(profiler.format_table(), flush=True)
            scenario_summaries.append(
                {
                    "scenario": scenario_name,
                    "algo": algo,
                    "policy_architecture": policy_architecture,
                    "action_mode": config.environment.action_mode,
                    "device": str(device),
                    "run_dirs": [str(path) for path in run_dirs],
                    "timing_json": str(timing_json),
                    "timing_csv": str(timing_csv),
                    "total_runtime_seconds": profiler.total_runtime_seconds(),
                    "sections": profiler.sorted_rows(),
                }
            )

    _write_combined_summary(output_root, scenario_summaries)
    print(f"\n[profile] output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
