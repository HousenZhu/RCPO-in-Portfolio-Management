from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rcpo_portfolio.config import ProjectConfig, load_config
from rcpo_portfolio.market import generate_market_splits
from rcpo_portfolio.trainer import resume_experiment, run_experiment


def tiny_config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig()
    config.experiment.output_root = str(tmp_path / "runs")
    config.experiment.run_name = "tiny"
    config.experiment.seeds = [0]
    config.market.lookback = 5
    config.market.train_steps = 320
    config.market.validation_steps = 220
    config.market.test_steps = 220
    config.environment.episode_length = 32
    config.optimization.total_updates = 2
    config.optimization.rollout_steps = 128
    config.optimization.epochs = 2
    config.optimization.minibatch_size = 32
    config.ppo.total_updates = 2
    config.ppo.rollout_steps = 128
    config.ppo.epochs = 2
    config.ppo.minibatch_size = 32
    config.ppo.learning_rate = 1e-4
    config.ppo.learning_rate_final = 5e-5
    config.ppo.target_kl = 10.0
    config.ppo.early_stop_patience = None
    config.evaluation.episodes = 2
    config.rcpo.calibration_episodes = 2
    return config


def test_short_training_runs_for_rcpo_and_ppo(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    rcpo_runs = []
    for mode in ["downside", "sortino"]:
        config.experiment.run_name = f"tiny_{mode}"
        config.rcpo.constraint_mode = mode
        config.environment.active_constraint_preset = "c2"
        rcpo_runs.append(run_experiment(config, algo="rcpo")[0])
    config.experiment.run_name = "tiny_ppo"
    config.rcpo.constraint_mode = "downside"
    config.environment.active_constraint_preset = "c2"
    ppo_run = run_experiment(config, algo="ppo_unconstrained")[0]
    rcpo_run = rcpo_runs[0]
    sortino_run = rcpo_runs[1]

    assert (rcpo_run / "checkpoint_last.pt").exists()
    assert (rcpo_run / "checkpoint_best.pt").exists()
    assert (sortino_run / "checkpoint_last.pt").exists()
    assert (sortino_run / "checkpoint_best.pt").exists()
    assert (rcpo_run / "metrics.jsonl").exists()
    assert (rcpo_run / "evaluation" / "training_return.png").exists()
    assert (rcpo_run / "evaluation" / "training_turnover.png").exists()
    assert (rcpo_run / "evaluation_best" / "summary_validation.json").exists()
    assert (rcpo_run / "evaluation_best" / "summary_test.json").exists()
    assert (rcpo_run / "evaluation_last" / "summary_validation.json").exists()
    assert (rcpo_run / "evaluation_last" / "summary_test.json").exists()
    assert not (rcpo_run / "evaluation" / "summary_test.json").exists()
    assert (ppo_run / "checkpoint_last.pt").exists()
    assert (ppo_run / "checkpoint_best.pt").exists()
    assert (ppo_run / "evaluation_best" / "summary_test.json").exists()
    with (rcpo_run / "evaluation_best" / "summary_test.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "average_constraint_cost" in payload
    assert "average_group_a_min_violation_cost" in payload
    assert "average_group_a_weight" in payload
    assert (rcpo_run / "evaluation_best" / "mean_cumulative_return_test.png").exists()
    assert (rcpo_run / "evaluation_best" / "cumulative_return_test.png").exists()
    with (rcpo_run / "config_snapshot.yaml").open("r", encoding="utf-8") as handle:
        snapshot = handle.read()
    assert "active_constraint_preset: c2" in snapshot
    assert "constraint_mode: downside" in snapshot
    assert "resolved_group_a_min_weight: 0.25" in snapshot
    with (sortino_run / "config_snapshot.yaml").open("r", encoding="utf-8") as handle:
        sortino_snapshot = handle.read()
    assert "constraint_mode: sortino" in sortino_snapshot
    with (ppo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        ppo_metric = json.loads(handle.readline())
    assert "approx_kl" in ppo_metric
    assert "clip_fraction" in ppo_metric
    assert "learning_rate" in ppo_metric
    assert "validation_annualized_return" in ppo_metric
    assert ppo_metric["learning_rate"] == config.ppo.learning_rate
    with (rcpo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rcpo_metric = json.loads(handle.readline())
    assert rcpo_metric["learning_rate"] == config.optimization.learning_rate
    assert "validation_turnover" in rcpo_metric
    assert "turnover_cap" in rcpo_metric

    original_metric_count = sum(1 for _ in (ppo_run / "metrics.jsonl").open("r", encoding="utf-8"))
    config.ppo.total_updates = 1
    resumed_run = resume_experiment(config, algo="ppo_unconstrained", run_dir=ppo_run)
    resumed_metric_count = sum(1 for _ in (resumed_run / "metrics.jsonl").open("r", encoding="utf-8"))
    assert resumed_metric_count == original_metric_count + 1


def test_load_config_supports_validation_and_legacy_train_test_only(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "market:",
                "  train_steps: 123",
                "  test_steps: 78",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.market.train_steps == 123
    assert config.market.test_steps == 78
    assert config.market.validation_steps == ProjectConfig().market.validation_steps

    config_path.write_text(
        "\n".join(
            [
                "market:",
                "  train_steps: 123",
                "  validation_steps: 456",
                "  test_steps: 78",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.market.validation_steps == 456


def test_market_splits_include_distinct_validation_and_test() -> None:
    config = ProjectConfig().market
    config.lookback = 2
    config.train_steps = 12
    config.validation_steps = 10
    config.test_steps = 8

    splits = generate_market_splits(config, seed=5)

    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"].risky_returns.shape[0] == 14
    assert splits["validation"].risky_returns.shape[0] == 12
    assert splits["test"].risky_returns.shape[0] == 10
    np.testing.assert_allclose(
        splits["validation"].risky_returns[: config.lookback],
        splits["train"].risky_returns[-config.lookback :],
    )
    np.testing.assert_allclose(
        splits["test"].risky_returns[: config.lookback],
        splits["train"].risky_returns[-config.lookback :],
    )
    assert not np.array_equal(
        splits["validation"].risky_returns[config.lookback :],
        splits["test"].risky_returns[config.lookback : config.lookback + config.test_steps],
    )
