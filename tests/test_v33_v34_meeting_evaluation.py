from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_v33_v34_meeting import (
    RolloutCache,
    build_branches,
    checkpoint_rows,
    discover_specs,
    paired_effects,
)
from rcpo_portfolio.config import load_config
from rcpo_portfolio.market import generate_continuation_split


def test_meeting_manifest_discovers_merged_v34_runs_and_controls() -> None:
    specs = discover_specs()
    v34 = [spec for spec in specs if spec.version == "v3.4"]
    assert len(specs) == 47
    assert len({spec.run_dir.resolve() for spec in specs}) == 45
    assert len(v34) == 24
    assert sum(spec.spec_id.endswith("_control") for spec in v34) == 2
    assert {spec.seed for spec in v34 if spec.condition_id == "v34_standalone_m2_e3"} == {0, 1, 2}
    rows = checkpoint_rows(specs)
    assert all(row["best_return_available"] for row in rows)
    assert any(not row["robust_feasible_available"] for row in rows)


def test_rollout_cache_round_trip(tmp_path: Path) -> None:
    cache = RolloutCache(tmp_path / "cache.sqlite")
    try:
        returns = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
        metrics = {"max_drawdown": 0.02, "turnover": 0.1}
        cache.put("key", returns, metrics)
        cached = cache.get("key")
        assert cached is not None
        np.testing.assert_array_equal(cached[0], returns)
        assert cached[1] == metrics
    finally:
        cache.close()


def test_canonical_holdout_future_is_identical_across_windows() -> None:
    reference = load_config(ROOT / "configs" / "v3.4_experiment2_8assets" / "base.yaml")
    branches = build_branches(reference, [91_000], 1)
    _anchor_index, _anchor_seed, _branch_index, future_seed, anchor, canonical = branches[0]
    for lookback in (20, 30, 40, 60):
        config = load_config(
            ROOT / "configs" / "v3.4_experiment2_8assets" / f"window_{lookback}.yaml"
        )
        market = generate_continuation_split(config.market, anchor, 252, future_seed)
        np.testing.assert_array_equal(market.risky_returns[-252:], canonical.risky_returns[-252:])


def test_paired_effects_match_seed_anchor_and_branch() -> None:
    rows = []
    for condition, base in (("v34_standalone_m2_e3", 0.01), ("v34_standalone_m8_e3", 0.03)):
        for seed in (0, 1):
            for branch in (0, 1):
                rows.append({
                    "condition_id": condition,
                    "checkpoint_type": "best_return",
                    "seed": seed,
                    "anchor_index": 0,
                    "branch_index": branch,
                    "relative_wealth": base + 0.001 * seed,
                    "max_drawdown": 0.10,
                    "common_feasible": 1.0,
                    "average_turnover": 0.02,
                })
    effects = [row for row in paired_effects(rows) if row["comparison"] == "market_2_to_8_e3"]
    assert len(effects) == 2
    assert all(row["paired_markets"] == 2 for row in effects)
    assert all(row["mean_relative_wealth_delta"] == pytest.approx(0.02) for row in effects)
