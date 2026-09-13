from __future__ import annotations

import argparse
import sys
from pathlib import Path

import compare_section9_simplex as comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a softmax RCPO allocation-penalty baseline against simplex policies."
        )
    )
    parser.add_argument(
        "--allocation-run-dir",
        required=True,
        help="Seed directory for the trained RCPO allocation-penalty run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            comparison.PROJECT_ROOT
            / "evaluation"
            / "allocation_penalty_vs_simplex_comparison"
        ),
        help="Folder where comparison PNG/CSV/JSON files are written.",
    )
    parser.add_argument(
        "--future-market-count",
        type=int,
        default=20,
        help="Number of shared continuation markets to evaluate.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    root = comparison.PROJECT_ROOT
    comparison.DEFAULT_POLICIES = [
        comparison.PolicySpec(
            label="PPO Gaussian Simplex v2.2",
            run_dir=root
            / "runs"
            / "simplex_v2.2_ppo_unconstrained_none_20260621_143156"
            / "seed_0",
        ),
        comparison.PolicySpec(
            label="RCPO Allocation Penalty",
            run_dir=Path(args.allocation_run_dir),
        ),
        comparison.PolicySpec(
            label="PPO Dirichlet Simplex v2.2",
            run_dir=root
            / "runs"
            / "simplex_v2.2_ppo_unconstrained_none_20260621_162247"
            / "seed_0",
        ),
        comparison.PolicySpec(
            label="RCPO Gaussian Simplex v2.2",
            run_dir=root
            / "runs"
            / "simplex_v2.2_rcpo_none_20260621_143215"
            / "seed_0",
        ),
        comparison.PolicySpec(
            label="RCPO Dirichlet Simplex v2.2",
            run_dir=root
            / "runs"
            / "simplex_v2.2_rcpo_none_20260621_143223"
            / "seed_0",
        ),
    ]
    sys.argv = [
        "compare_section9_simplex.py",
        "--output-dir",
        str(args.output_dir),
        "--future-market-count",
        str(args.future_market_count),
    ]
    comparison.main()


if __name__ == "__main__":
    main()
