from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.trainer import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Short V3.4 training and runtime smoke test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--output-root", default="profile_outputs/v34_smoke")
    parser.add_argument("--validation-branches", type=int, default=None)
    parser.add_argument("--test-branches", type=int, default=None)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    config = load_config(args.config)
    config.experiment.seeds = [0]
    config.experiment.run_name = f"smoke_{config.experiment.run_name}"
    config.optimization.total_updates = args.updates
    if args.rollout_steps is not None:
        config.optimization.rollout_steps = args.rollout_steps
    config.optimization.learning_rate_final = config.optimization.learning_rate
    config.optimization.early_stop_patience = None
    config.runtime.device = "cpu"
    if args.validation_branches is not None:
        config.evaluation.validation_branch_count = args.validation_branches
    if args.test_branches is not None:
        config.evaluation.test_branch_count = args.test_branches
    sync_rcpo_constraint_settings(config)

    start = time.perf_counter()
    run_dirs = run_experiment(config, "rcpo", output_root=args.output_root)
    elapsed = time.perf_counter() - start
    result = {
        "config": args.config,
        "updates": args.updates,
        "rollout_steps": config.optimization.rollout_steps,
        "validation_branches": config.evaluation.validation_branch_count,
        "test_branches": config.evaluation.test_branch_count,
        "elapsed_seconds": elapsed,
        "seconds_per_update": elapsed / args.updates,
        "projected_60000_update_hours_with_30pct_buffer": (
            elapsed / args.updates * 60_000 * 1.30 / 3600
        ),
        "run_dir": str(run_dirs[0]),
    }
    output = Path(args.output_root) / f"{config.experiment.run_name}_runtime_estimate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
