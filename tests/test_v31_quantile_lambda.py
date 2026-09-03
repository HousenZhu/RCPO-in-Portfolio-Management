from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.trainer import run_experiment


@pytest.mark.parametrize(
    ("config_name", "lambda_lr_up"),
    [
        ("pilot_q80_lr025.yaml", 0.00025),
        ("pilot_q80_lr040.yaml", 0.00040),
    ],
)
def test_v31_pilot_configs_use_episode_quantile_gap(
    config_name: str,
    lambda_lr_up: float,
) -> None:
    config = load_config(Path("configs/v3.1_experiment2_8assets") / config_name)
    sync_rcpo_constraint_settings(config)

    assert config.experiment.run_name.startswith("simplex_v3.1_")
    assert config.market.num_risky_assets == 8
    assert config.optimization.total_updates == 14_000
    assert config.rcpo.lambda_gap_mode == "episode_quantile"
    assert config.rcpo.constraint_quantile == pytest.approx(0.80)
    assert config.rcpo.constraint_gap_window_episodes == 64
    assert config.rcpo.constraint_gap_min_episodes == 16
    assert config.rcpo.lambda_lr_up == pytest.approx(lambda_lr_up)
    assert config.rcpo.lambda_lr_down == pytest.approx(0.005)
    assert config.environment.benchmark_drawdown_margin == pytest.approx(0.90)
    assert config.environment.drawdown_budget_floor == pytest.approx(0.05)
    assert config.environment.drawdown_cost_scale == pytest.approx(0.10)
    assert config.evaluation.validation_interval_updates == 100
    assert config.logging.live_validation_plot is False


def test_v31_quantile_smoke_logs_and_checkpoints_gap_state(tmp_path: Path) -> None:
    config = load_config(
        "configs/v3.1_experiment2_8assets/pilot_q80_lr025.yaml"
    )
    config.experiment.output_root = str(tmp_path)
    config.experiment.run_name = "tiny_v31"
    config.market.lookback = 2
    config.market.train_market_count = 2
    config.market.train_steps = 40
    config.market.validation_steps = 8
    config.market.test_steps = 8
    config.environment.episode_length = 8
    config.optimization.total_updates = 1
    config.optimization.rollout_steps = 32
    config.optimization.epochs = 1
    config.optimization.minibatch_size = 16
    config.rcpo.constraint_gap_window_episodes = 8
    config.rcpo.constraint_gap_min_episodes = 2
    config.evaluation.validation_branch_count = 2
    config.evaluation.test_branch_count = 2
    config.evaluation.validation_interval_updates = 1

    run_dir = run_experiment(
        config,
        algo="rcpo",
        disable_artifacts=True,
    )[0]
    metric = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    checkpoint = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")

    assert metric["batch_completed_episode_count"] == 4
    assert metric["lambda_gap_mode"] == "episode_quantile"
    assert metric["lambda_gap_effective_mode"] == "episode_quantile"
    assert metric["lambda_gap_window_episode_count"] == 4
    assert metric["lambda_gap"] == pytest.approx(
        metric["lambda_gap_window_quantile"]
    )
    assert 0.0 <= metric["lambda_gap_window_feasible_rate"] <= 1.0
    assert checkpoint["lambda_gap_mode"] == "episode_quantile"
    assert checkpoint["constraint_quantile"] == pytest.approx(0.80)
    assert len(checkpoint["constraint_gap_history"]) == 4

