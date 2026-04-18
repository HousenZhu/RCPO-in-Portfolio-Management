from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.trainer import resume_experiment, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCPO portfolio management experiments.")
    parser.add_argument(
        "--algo",
        required=True,
        choices=["rcpo", "ppo_unconstrained", "equal_weight"],
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the experiment output directory.",
    )
    parser.add_argument(
        "--constraint-preset",
        choices=["c1", "c2", "c3"],
        default=None,
        help="Optional override for the active long-only group-constraint preset.",
    )
    constraint_group = parser.add_mutually_exclusive_group()
    constraint_group.add_argument(
        "--constraint-downside",
        action="store_true",
        help="Use normalized downside semivariance as the RCPO constraint cost.",
    )
    constraint_group.add_argument(
        "--constraint-sortino",
        action="store_true",
        help="Use Sortino target-violation cost as the RCPO constraint cost.",
    )
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Seed run directory to continue, for example runs/latest_ppo.../seed_0.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="checkpoint_last.pt",
        help="Checkpoint file inside --resume-run-dir to continue from.",
    )
    args = parser.parse_args()
    has_constraint_flag = args.constraint_downside or args.constraint_sortino
    if args.algo == "rcpo" and not has_constraint_flag:
        parser.error("RCPO requires exactly one of --constraint-downside or --constraint-sortino.")
    if args.algo != "rcpo" and has_constraint_flag:
        parser.error("--constraint-downside and --constraint-sortino are only valid with --algo rcpo.")
    return args


def main() -> None:
    args = parse_args()
    resume_run_dir = Path(args.resume_run_dir) if args.resume_run_dir is not None else None
    config_path = (
        resume_run_dir / "config_snapshot.yaml"
        if resume_run_dir is not None
        else Path(args.config)
    )
    config = load_config(config_path)
    if args.constraint_preset is not None:
        config.environment.active_constraint_preset = args.constraint_preset
    if args.algo == "rcpo":
        config.rcpo.constraint_mode = "sortino" if args.constraint_sortino else "downside"
    sync_rcpo_constraint_settings(config)
    if resume_run_dir is not None:
        run_dir = resume_experiment(
            config=config,
            algo=args.algo,
            run_dir=resume_run_dir,
            checkpoint_name=args.resume_checkpoint,
        )
        print(run_dir)
    else:
        run_dirs = run_experiment(config=config, algo=args.algo, output_root=args.output_root)
        for run_dir in run_dirs:
            print(run_dir)


if __name__ == "__main__":
    main()
