from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rcpo_portfolio.config import ProjectConfig, load_config
from rcpo_portfolio.market import generate_market_splits
from rcpo_portfolio.trainer import resume_experiment, run_experiment


def tiny_config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig()
    config.experiment.output_root = str(tmp_path / "runs")
    config.experiment.run_name = "tiny"
    config.experiment.seeds = [0]
    config.runtime.device = "cpu"
    config.market.lookback = 5
    config.market.train_market_count = 2
    config.market.train_steps = 160
    config.market.validation_steps = 96
    config.market.test_steps = 96
    config.environment.episode_length = 32
    config.environment.drawdown_budget_floor = 0.02
    config.environment.benchmark_drawdown_margin = 0.90
    config.environment.drawdown_cost_scale = 0.01
    config.rcpo.alpha = None
    config.rcpo.alpha_budget_ratio = 0.05
    config.rcpo.lambda_lr_up = 0.015
    config.rcpo.lambda_lr_down = 0.03
    config.optimization.total_updates = 1
    config.optimization.rollout_steps = 64
    config.optimization.epochs = 1
    config.optimization.minibatch_size = 32
    config.optimization.target_kl = 10.0
    config.optimization.early_stop_patience = None
    config.ppo.total_updates = 1
    config.ppo.rollout_steps = 64
    config.ppo.epochs = 1
    config.ppo.minibatch_size = 32
    config.ppo.learning_rate = 1e-4
    config.ppo.learning_rate_final = 5e-5
    config.ppo.target_kl = 10.0
    config.ppo.early_stop_patience = None
    config.evaluation.episodes = 1
    config.evaluation.validation_branch_count = 2
    config.evaluation.test_branch_count = 2
    config.reward_correction.hidden_sizes = [16]
    config.reward_correction.train_epochs_per_update = 1
    config.reward_correction.num_bins = 5
    config.reward_correction.gdrc_num_candidates = 2
    config.reward_correction.gdrc_range_window_updates = 2
    return config


def test_short_training_runs_for_rcpo_and_ppo(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    runs: dict[tuple[str, str], Path] = {}
    for reward_mode in ["none", "drc", "gdrc"]:
        config.reward_correction.mode = reward_mode
        config.experiment.run_name = f"tiny_rcpo_{reward_mode}"
        runs[("rcpo", reward_mode)] = run_experiment(config, algo="rcpo")[0]
        config.experiment.run_name = f"tiny_ppo_{reward_mode}"
        runs[("ppo_unconstrained", reward_mode)] = run_experiment(
            config, algo="ppo_unconstrained"
        )[0]

    rcpo_run = runs[("rcpo", "none")]
    ppo_run = runs[("ppo_unconstrained", "none")]
    rcpo_drc_run = runs[("rcpo", "drc")]
    rcpo_gdrc_run = runs[("rcpo", "gdrc")]
    ppo_drc_run = runs[("ppo_unconstrained", "drc")]
    ppo_gdrc_run = runs[("ppo_unconstrained", "gdrc")]

    assert (rcpo_run / "checkpoint_last.pt").exists()
    assert (rcpo_run / "checkpoint_best.pt").exists()
    assert (rcpo_drc_run / "checkpoint_best.pt").exists()
    assert (rcpo_gdrc_run / "checkpoint_best.pt").exists()
    assert (ppo_run / "checkpoint_last.pt").exists()
    assert (ppo_run / "checkpoint_best.pt").exists()
    assert (ppo_drc_run / "checkpoint_best.pt").exists()
    assert (ppo_gdrc_run / "checkpoint_best.pt").exists()

    assert (rcpo_run / "metrics.jsonl").exists()
    assert (rcpo_run / "evaluation" / "group_weights_validation.png").exists()
    assert (rcpo_run / "evaluation" / "training_return.png").exists()
    assert (rcpo_run / "evaluation" / "training_turnover.png").exists()
    assert (rcpo_run / "evaluation_best" / "summary_validation.json").exists()
    assert (rcpo_run / "evaluation_best" / "summary_test.json").exists()
    assert (rcpo_run / "evaluation_last" / "summary_validation.json").exists()
    assert (rcpo_run / "evaluation_last" / "summary_test.json").exists()
    assert not (rcpo_run / "evaluation" / "summary_test.json").exists()
    assert (rcpo_run / "evaluation_best" / "mean_cumulative_return_test.png").exists()
    assert (rcpo_run / "evaluation_best" / "cumulative_return_test.png").exists()
    assert (rcpo_run / "evaluation_best" / "drawdown_test.png").exists()
    assert (rcpo_run / "evaluation_best" / "drawdown_constraint_cost_test.png").exists()
    assert (ppo_run / "evaluation_best" / "summary_test.json").exists()

    with (rcpo_run / "evaluation_best" / "summary_test.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "average_constraint_cost" in payload
    assert "benchmark_max_drawdown" in payload
    assert "effective_drawdown_budget" in payload
    assert "average_drawdown_gap" in payload
    assert "average_drawdown_violation" in payload
    assert "average_drawdown_constraint_cost" in payload
    assert "average_group_a_min_violation_cost" in payload
    assert "average_group_a_weight" in payload

    with (rcpo_run / "config_snapshot.yaml").open("r", encoding="utf-8") as handle:
        snapshot = handle.read()
    assert "active_constraint_preset: c2" in snapshot
    assert "constraint_mode: max_drawdown" in snapshot
    assert "drawdown_budget_floor: 0.02" in snapshot
    assert "benchmark_drawdown_margin: 0.9" in snapshot
    assert "drawdown_cost_scale: 0.01" in snapshot
    assert "alpha_budget_ratio: 0.05" in snapshot
    assert "lambda_lr_up: 0.015" in snapshot
    assert "lambda_lr_down: 0.03" in snapshot
    assert "runtime:" in snapshot
    assert "device: cpu" in snapshot
    assert "resolved_group_a_min_weight: 0.25" in snapshot

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
    assert rcpo_metric["constraint_mode"] == "max_drawdown"
    assert rcpo_metric["alpha_mode"] == "budget_ratio"
    assert rcpo_metric["alpha_budget_ratio"] == config.rcpo.alpha_budget_ratio
    assert rcpo_metric["lambda_lr_up"] == config.rcpo.lambda_lr_up
    assert rcpo_metric["lambda_lr_down"] == config.rcpo.lambda_lr_down
    assert "lambda_gap" in rcpo_metric
    assert rcpo_metric["alpha"] == rcpo_metric["batch_alpha_target_mean"]
    assert rcpo_metric["alpha"] > 0.0
    assert "validation_turnover" in rcpo_metric
    assert "validation_mean_excess_cumulative_return" in rcpo_metric
    assert "validation_win_rate_vs_equal_weight" in rcpo_metric
    assert "validation_max_drawdown" in rcpo_metric
    assert "validation_benchmark_max_drawdown" in rcpo_metric
    assert "validation_effective_drawdown_budget" in rcpo_metric
    assert "validation_alpha_target" in rcpo_metric
    assert "validation_drawdown_gap" in rcpo_metric
    assert "validation_drawdown_violation" in rcpo_metric
    assert "validation_drawdown_constraint_cost" in rcpo_metric
    assert "batch_current_drawdown_mean" in rcpo_metric
    assert "batch_max_drawdown_mean" in rcpo_metric
    assert "batch_benchmark_max_drawdown_mean" in rcpo_metric
    assert "batch_effective_drawdown_budget_mean" in rcpo_metric
    assert "batch_alpha_target_mean" in rcpo_metric
    assert "batch_drawdown_gap_mean" in rcpo_metric
    assert "batch_drawdown_violation_mean" in rcpo_metric
    assert "batch_concentration_mean" in rcpo_metric
    assert "batch_diversification_cost_mean" in rcpo_metric
    assert rcpo_metric["validation_evaluated"] == 1
    assert rcpo_metric["validation_interval_updates"] == config.evaluation.validation_interval_updates
    assert rcpo_metric["device"] == "cpu"
    assert rcpo_metric["reward_correction_mode"] == "none"

    with (rcpo_drc_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rcpo_drc_metric = json.loads(handle.readline())
    assert rcpo_drc_metric["reward_correction_mode"] == "drc"
    assert "reward_correction_delta_abs_mean" in rcpo_drc_metric
    assert (rcpo_drc_run / "evaluation" / "training_reward_correction.png").exists()

    with (rcpo_gdrc_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rcpo_gdrc_metric = json.loads(handle.readline())
    assert rcpo_gdrc_metric["reward_correction_mode"] == "gdrc"
    assert rcpo_gdrc_metric["gdrc_selected_bins"] in {2, 4}
    assert (rcpo_gdrc_run / "evaluation" / "gdrc_selected_bins.png").exists()

    original_metric_count = sum(1 for _ in (ppo_run / "metrics.jsonl").open("r", encoding="utf-8"))
    config.reward_correction.mode = "none"
    config.ppo.total_updates = 1
    resumed_run = resume_experiment(config, algo="ppo_unconstrained", run_dir=ppo_run)
    resumed_metric_count = sum(1 for _ in (resumed_run / "metrics.jsonl").open("r", encoding="utf-8"))
    assert resumed_metric_count == original_metric_count + 1


def test_resume_rejects_legacy_rcpo_constraint_mode(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.reward_correction.mode = "none"
    config.experiment.run_name = "tiny_rcpo_resume_guard"
    run_dir = run_experiment(config, algo="rcpo")[0]

    checkpoint_path = run_dir / "checkpoint_legacy.pt"
    payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    payload["constraint_semantics"] = "fixed_drawdown_v0"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="Legacy fixed-budget drawdown checkpoints are not supported"):
        resume_experiment(
            config,
            algo="rcpo",
            run_dir=run_dir,
            checkpoint_name="checkpoint_legacy.pt",
        )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_short_training_can_use_cuda_when_configured(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.runtime.device = "cuda"
    config.reward_correction.mode = "none"
    config.experiment.run_name = "tiny_cuda_rcpo"

    run_dir = run_experiment(config, algo="rcpo")[0]

    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        metric = json.loads(handle.readline())
    assert metric["device"] == "cuda"
    checkpoint = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    assert checkpoint["device"] == "cuda"
