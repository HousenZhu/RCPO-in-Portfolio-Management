from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_v34 import (
    RunSpec,
    common_drawdown_metrics,
    compare,
    drawdown_from_initial_wealth,
    own_drawdown_metrics,
)
from compare_section9_simplex import rollout_policy
from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.market import (
    generate_continuation_split,
    generate_market_split,
    generate_train_markets,
    resolve_continuation_anchor,
)
from rcpo_portfolio.trainer import resume_experiment, run_experiment

CONFIG_DIR = ROOT / "configs" / "v3.4_experiment2_8assets"
V33_DIR = ROOT / "configs" / "v3.3_experiment2_8assets"


def test_v34_manifest_has_22_unique_runs() -> None:
    with (CONFIG_DIR / "tasks.tsv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    assert len(tasks) == 22
    assert len({task["task_id"] for task in tasks}) == 22
    assert len({(task["config"], task["seed"]) for task in tasks}) == 22
    assert ("base.yaml", "0") not in {(task["config"], task["seed"]) for task in tasks}
    assert ("standalone_m8_e3.yaml", "0") not in {
        (task["config"], task["seed"]) for task in tasks
    }
    for task in tasks:
        config = load_config(CONFIG_DIR / task["config"])
        assert config.optimization.total_updates == 60_000
        assert config.environment.episode_length == 252
        assert config.market.train_steps == 5_040
        assert config.evaluation.continuation_seed_mode == "fixed_offset"
    assert sum(task["group"] == "factorial" for task in tasks) == 10
    assert sum(task["group"] == "counterfactual" for task in tasks) == 6
    assert sum(task["group"] == "window" for task in tasks) == 4
    assert sum(task["group"] == "margin" for task in tasks) == 2


def test_v33_seed_zero_paths_are_preserved() -> None:
    old = load_config(V33_DIR / "rcpo_gaussian_m2_s5040_e3.yaml")
    new = load_config(CONFIG_DIR / "base.yaml")
    old_train = generate_train_markets(old.market, 0)
    new_train = generate_train_markets(new.market, 0)
    for left, right in zip(old_train, new_train, strict=True):
        np.testing.assert_array_equal(left.risky_returns, right.risky_returns)
    old_anchor = resolve_continuation_anchor(
        old.market, old_train[0], mode="fixed_market", fixed_seed=90_000, fixed_steps=5_040
    )
    new_anchor = resolve_continuation_anchor(
        new.market, new_train[0], mode="fixed_market", fixed_seed=90_000,
        fixed_steps=5_040, fixed_generation_lookback=20,
    )
    np.testing.assert_array_equal(old_anchor.risky_returns, new_anchor.risky_returns)
    for offset in (100_000, 200_000):
        old_future = generate_continuation_split(old.market, old_anchor, 252, offset)
        new_future = generate_continuation_split(new.market, new_anchor, 252, offset)
        np.testing.assert_array_equal(old_future.risky_returns, new_future.risky_returns)


def test_window_training_tradable_paths_are_identical() -> None:
    windows = [load_config(CONFIG_DIR / f"window_{value}.yaml") for value in (20, 30, 40, 60)]
    for market_index in range(2):
        paths = [generate_train_markets(config.market, 0)[market_index] for config in windows]
        for config, market in zip(windows, paths, strict=True):
            assert len(market.risky_returns) == 5_040 + config.market.lookback
            np.testing.assert_array_equal(
                market.risky_returns[config.market.lookback :],
                paths[0].risky_returns[windows[0].market.lookback :],
            )


def test_window_future_paths_and_episode_lengths_match() -> None:
    configs = [load_config(CONFIG_DIR / f"window_{value}.yaml") for value in (20, 30, 40, 60)]
    anchor = generate_market_split(configs[0].market, 5_040, 90_000)
    futures = [generate_continuation_split(config.market, anchor, 252, 100_000) for config in configs]
    for config, market in zip(configs, futures, strict=True):
        np.testing.assert_array_equal(market.risky_returns[-252:], futures[0].risky_returns[-252:])
        env = PortfolioEnv(config.environment, market, config.market)
        obs, _ = env.reset(options={"start_index": config.market.lookback})
        count = 0
        while True:
            obs, _reward, terminated, truncated, _info = env.step(env.neutral_action())
            count += 1
            if terminated or truncated:
                break
        assert count == 252


def test_fixed_continuation_seeds_ignore_train_seed() -> None:
    config = load_config(CONFIG_DIR / "base.yaml")
    anchors = []
    futures = []
    for seed in (0, 1, 2):
        train = generate_train_markets(config.market, seed)[0]
        anchor = resolve_continuation_anchor(
            config.market, train, mode=config.evaluation.continuation_anchor_mode,
            fixed_seed=config.evaluation.fixed_anchor_seed,
            fixed_steps=config.evaluation.fixed_anchor_steps,
            fixed_generation_lookback=config.evaluation.fixed_anchor_generation_lookback,
        )
        anchors.append(anchor)
        futures.append(generate_continuation_split(
            config.market, anchor, 252, config.evaluation.validation_seed_offset
        ))
    for anchor in anchors[1:]:
        np.testing.assert_array_equal(anchor.risky_returns, anchors[0].risky_returns)
    for market in futures[1:]:
        np.testing.assert_array_equal(market.risky_returns, futures[0].risky_returns)


def test_margin_configs_only_change_margin() -> None:
    base = load_config(CONFIG_DIR / "standalone_m2_e4.yaml")
    for name, margin in (("margin_085.yaml", 0.85), ("margin_080.yaml", 0.80)):
        config = load_config(CONFIG_DIR / name)
        assert config.environment.benchmark_drawdown_margin == margin
        config.environment.benchmark_drawdown_margin = 0.90
        config.experiment.run_name = base.experiment.run_name
        assert config.to_dict() == base.to_dict()


def test_common_drawdown_uses_initial_wealth_peak_and_fixed_margin() -> None:
    agent = np.asarray([-0.04, 0.01, -0.02, 0.03])
    baseline = np.asarray([-0.02, 0.01, -0.01, 0.02])
    assert drawdown_from_initial_wealth(agent)[0] == pytest.approx(0.04)
    default = common_drawdown_metrics(agent, baseline)
    strict = common_drawdown_metrics(agent, baseline, margin=0.80)
    assert default["cost"] >= 0
    assert strict["cost"] >= default["cost"]


def test_common_margin_090_matches_environment_cost() -> None:
    config = load_config(CONFIG_DIR / "standalone_m2_e4.yaml")
    market = generate_market_split(config.market, 252, 67890)
    env = PortfolioEnv(config.environment, market, config.market)
    result = rollout_policy(env, lambda _obs: env.neutral_action())
    returns = np.asarray(result["returns"])
    common = common_drawdown_metrics(returns, returns)
    own = own_drawdown_metrics(config, result)
    assert common["cost"] == pytest.approx(own["cost"], abs=1e-7)
    assert common["alpha"] == pytest.approx(own["alpha"], abs=1e-7)


def test_invalid_generation_settings_rejected() -> None:
    config = load_config(CONFIG_DIR / "window_60.yaml")
    config.market.train_generation_lookback = 40
    with pytest.raises(ValueError, match="train_generation_lookback"):
        sync_rcpo_constraint_settings(config)
    config = load_config(CONFIG_DIR / "base.yaml")
    config.evaluation.continuation_seed_mode = "unknown"
    with pytest.raises(ValueError, match="continuation_seed_mode"):
        sync_rcpo_constraint_settings(config)


def test_v34_absolute_resume_without_extra_validation(tmp_path: Path) -> None:
    config = load_config(CONFIG_DIR / "base.yaml")
    config.experiment.output_root = str(tmp_path)
    config.market.lookback = 5
    config.market.train_steps = 40
    config.market.validation_steps = 8
    config.market.test_steps = 8
    config.environment.episode_length = 8
    config.evaluation.fixed_anchor_steps = 40
    config.evaluation.validation_branch_count = 1
    config.evaluation.test_branch_count = 1
    config.optimization.total_updates = 1
    config.optimization.rollout_steps = 16
    config.optimization.epochs = 1
    config.optimization.minibatch_size = 16
    config.optimization.early_stop_patience = None
    config.logging.live_validation_plot = False
    run_dir = run_experiment(config, algo="rcpo", disable_artifacts=True)[0]
    resume_experiment(
        config, algo="rcpo", run_dir=run_dir,
        target_total_updates=2, disable_artifacts=True,
    )
    assert len((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((run_dir / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    output_dir = tmp_path / "comparison"
    compare(
        [RunSpec("smoke_window_5", "base.yaml", 0, run_dir)],
        output_dir,
        anchor_seeds=[91_000],
        branch_count=1,
        evaluation_steps=8,
    )
    assert (output_dir / "branch_metrics.csv").exists()
    assert (output_dir / "factorial_relative_wealth.png").exists()
