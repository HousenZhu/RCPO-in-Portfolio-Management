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
        help="Optional override for the active allocation-constraint preset.",
    )
    parser.add_argument(
        "--constraint-drawdown",
        action="store_true",
        help="Use maximum drawdown target-violation cost as the RCPO constraint cost.",
    )
    reward_group = parser.add_mutually_exclusive_group()
    reward_group.add_argument(
        "--use-drc",
        action="store_true",
        help="Train with a distributional reward critic reward-correction layer.",
    )
    reward_group.add_argument(
        "--use-gdrc",
        action="store_true",
        help="Train with a general distributional reward critic ensemble.",
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
    if args.algo == "rcpo" and not args.constraint_drawdown:
        parser.error("RCPO requires --constraint-drawdown.")
    if args.algo != "rcpo" and args.constraint_drawdown:
        parser.error("--constraint-drawdown is only valid with --algo rcpo.")
    if args.algo == "equal_weight" and (args.use_drc or args.use_gdrc):
        parser.error("--use-drc and --use-gdrc are not valid with --algo equal_weight.")
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
        config.rcpo.constraint_mode = "max_drawdown"
    if args.use_drc:
        config.reward_correction.mode = "drc"
    elif args.use_gdrc:
        config.reward_correction.mode = "gdrc"
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
