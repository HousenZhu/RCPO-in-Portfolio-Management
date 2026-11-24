"""Run individual paper experiment tasks or build an evaluation manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcpo_portfolio.config import load_config
from rcpo_portfolio.trainer import resume_experiment, run_experiment


TASKS = (
    ("rcpo_allocation_penalty", "allocation_drawdown", "rcpo"),
    ("caosd_ppo", "ppo", "ppo_unconstrained"),
    ("caosd_rcpo_shared", "shared", "rcpo"),
    ("caosd_rcpo_standalone", "standalone", "rcpo"),
    ("caosd_rcpo_cf_r", "cf_r", "rcpo"),
    ("caosd_rcpo_cf_c", "cf_c", "rcpo"),
    ("caosd_rcpo_cf_rc", "cf_rc", "rcpo"),
)
SEEDS = (0, 1, 2)
CONFIGS = ROOT / "configs" / "paper_gaussian"


def config_for(name, algo, seed, smoke):
    config = load_config(CONFIGS / f"{name}.yaml")
    config.experiment.seeds = [seed]
    optimization = config.ppo if algo == "ppo_unconstrained" else config.optimization
    if smoke:
        optimization.total_updates = 1
        optimization.rollout_steps = 32
        optimization.minibatch_size = 16
        optimization.epochs = 1
        config.evaluation.validation_interval_updates = 1
        config.evaluation.validation_branch_count = 1
        config.evaluation.test_branch_count = 1
        config.logging.validation_checkpoint_interval_updates = 1
    return config, optimization


def find_run(run_root, name, seed):
    candidates = sorted(
        path
        for arm in (run_root / name, run_root / f"v4.1_{name}")
        for path in arm.glob(f"seed_{seed}/*/seed_{seed}")
    )
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous run for {name}, seed {seed}: {candidates}")
    return candidates[0] if candidates else None


def hardware_id():
    cpuinfo = Path("/proc/cpuinfo")
    cpu = platform.processor() or platform.machine()
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    threads = "/".join(os.environ.get(name, "unset") for name in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    return f"{platform.system()} | {cpu} | thread limits {threads}"


def peak_memory_bytes():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else 1024 * peak)


def train_task(run_root, index, smoke):
    if not 0 <= index < len(TASKS) * len(SEEDS):
        raise ValueError("Task index must be 0 through 20.")
    method_index, seed_index = divmod(index, len(SEEDS))
    method, name, algo = TASKS[method_index]
    seed = SEEDS[seed_index]
    config, optimization = config_for(name, algo, seed, smoke)
    if smoke:
        run_root /= "smoke"
    run = find_run(run_root, name, seed)
    if run is not None and (run / "training_summary.json").is_file():
        summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
        if summary["completed_updates"] == optimization.total_updates:
            print(f"Already completed: {run}")
            return
    fresh = run is None
    start = time.perf_counter()
    if fresh:
        run = run_experiment(
            config, algo, output_root=run_root / name / f"seed_{seed}", disable_artifacts=True
        )[0]
    else:
        config = load_config(run / "config_snapshot.yaml")
        resume_experiment(
            config, algo, run, disable_artifacts=True,
            target_total_updates=optimization.total_updates,
        )
    elapsed = time.perf_counter() - start
    summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    if summary["completed_updates"] != optimization.total_updates:
        raise RuntimeError(f"{run} did not complete its full training budget.")
    if fresh:
        memory = peak_memory_bytes()
        (run / "paper_compute.json").write_text(json.dumps({
            "algorithm_id": method, "training_seed": seed,
            "hardware_id": hardware_id(),
            "timing_protocol_id": "process_peak_rss_and_wall_with_validation_no_plots_v1",
            "training_real_transition_count": summary["retained_real_transition_count"],
            "optimizer_step_count": summary["retained_optimizer_step_count"],
            "shadow_transition_count": 0,
            "one_step_replacement_count": summary["retained_one_step_replacement_count"],
            "training_wall_time_seconds": [elapsed],
            "peak_training_memory_bytes": [memory] if memory is not None else [],
        }, indent=2), encoding="utf-8")
    print(run)


def build_manifest(run_root, output, smoke):
    runs, matched, compute = [], [], []
    if smoke:
        run_root /= "smoke"
    for method, name, algo in TASKS:
        for seed in ((0,) if smoke else SEEDS):
            config, optimization = config_for(name, algo, seed, smoke)
            run = find_run(run_root, name, seed)
            if run is None or not (run / "checkpoint_best_return.pt").is_file():
                raise FileNotFoundError(f"Missing best-return checkpoint for {method}, seed {seed}.")
            summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
            if summary["completed_updates"] != optimization.total_updates:
                raise ValueError(f"Incomplete training budget: {run}")
            interval = config.logging.validation_checkpoint_interval_updates
            entry = {
                "algorithm_id": method, "run_dir": str(run.resolve()), "training_seed": seed,
                "test_checkpoint": "checkpoint_best_return.pt",
                "validation_checkpoints": [
                    {"checkpoint": f"checkpoint_validation_{update:06d}.pt",
                     "training_real_transition_count": update * optimization.rollout_steps}
                    for update in range(0, optimization.total_updates + 1, interval)
                ],
            }
            runs.append(entry)
            if config.network.matched_branch_credit and algo == "rcpo":
                matched.append(entry)
            if (run / "paper_compute.json").is_file():
                compute.append(json.loads((run / "paper_compute.json").read_text(encoding="utf-8")))
    validation_count = 1 if smoke else 20
    test_anchors = (191000,) if smoke else (191000, 192000, 193000, 194000, 195000)
    test_count = 1 if smoke else 20
    payload = {
        "reference_config": str(CONFIGS / "base.yaml"),
        "actor_family": "gaussian",
        "protocol_id": "paper_gaussian_smoke" if smoke else "paper_gaussian_v1",
        "decoding": "deterministic",
        "validation": {
            "id": "paper_validation_anchor_90000",
            "markets": [
                {"id": f"val_{i:02d}", "anchor_seed": 90000, "continuation_seed": 100000 + i}
                for i in range(validation_count)
            ],
        },
        "test": {
            "id": "paper_heldout_anchors_191000_to_195000",
            "markets": [
                {"id": f"test_{a}_{i:02d}", "anchor_seed": seed,
                 "continuation_seed": 300000 + a * 1000 + i}
                for a, seed in enumerate(test_anchors)
                for i in range(test_count)
            ],
        },
        "runs": runs, "matched_runs": matched, "compute": compute,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--task", type=int, required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train_task(args.run_root.resolve(), args.task, args.smoke)
    else:
        build_manifest(args.run_root.resolve(), args.output, args.smoke)


if __name__ == "__main__":
    main()
