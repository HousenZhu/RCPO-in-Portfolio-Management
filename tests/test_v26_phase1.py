from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.trainer import RCPOTrainer, run_experiment
from tests.test_env import build_env


def test_relative_current_drawdown_recovers_while_max_drawdown_remains() -> None:
    env = build_env(
        constraint_mode="relative_current_drawdown",
        observation_schema_version=2,
        drawdown_budget_floor=0.05,
        drawdown_cost_scale=0.10,
        transaction_cost_bps=0.0,
    )
    observation, _ = env.reset(options={"start_index": 2})
    assert observation.shape[0] == env.observation_space.shape[0]
    assert observation[-7:].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.05, -0.05, 0.0]
    )

    loss = env._drawdown_components(-0.10, effective_drawdown_budget=0.05)
    assert loss["current_drawdown"] == pytest.approx(0.10)
    assert loss["max_drawdown"] == pytest.approx(0.10)
    assert loss["drawdown_violation"] == pytest.approx(0.05)
    assert loss["drawdown_constraint_cost"] == pytest.approx(0.025)

    assert env.state is not None
    env.state.portfolio_value = loss["portfolio_value"]
    env.state.running_peak_value = loss["running_peak_value"]
    env.state.current_drawdown = loss["current_drawdown"]
    env.state.max_drawdown = loss["max_drawdown"]
    recovered = env._drawdown_components(0.20, effective_drawdown_budget=0.05)
    assert recovered["current_drawdown"] == pytest.approx(0.0)
    assert recovered["max_drawdown"] == pytest.approx(0.10)
    assert recovered["drawdown_constraint_cost"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "config_name",
    [
        "simplex_ppo_gaussian.yaml",
        "simplex_rcpo_gaussian.yaml",
        "simplex_ppo_dirichlet.yaml",
        "simplex_rcpo_dirichlet.yaml",
        "rcpo_allocation_penalty.yaml",
        "rcpo_allocation_relative_drawdown_penalty.yaml",
    ],
)
def test_v26_configs_are_8_asset_60000_update_configs(config_name: str) -> None:
    config = load_config(Path("configs/v2.6_experiment2_8assets") / config_name)
    sync_rcpo_constraint_settings(config)
    optimization = config.ppo if "ppo_" in config_name else config.optimization
    assert config.market.num_risky_assets == 8
    assert optimization.total_updates == 60_000
    assert config.environment.observation_schema_version == 2
    assert config.logging.metrics_schema_version == 2
    assert config.environment.benchmark_drawdown_margin == pytest.approx(0.90)


def test_best_feasible_checkpoint_ranks_rate_before_validation_return() -> None:
    trainer = object.__new__(RCPOTrainer)
    trainer.config = SimpleNamespace(
        evaluation=SimpleNamespace(
            checkpoint_score="validation_mean_excess_cumulative_return"
        )
    )
    rows = [
        {
            "validation_annualized_return": 0.10,
            "validation_mean_excess_cumulative_return": 0.20,
            "validation_feasible_branch_rate": 0.90,
        },
        {
            "validation_annualized_return": 0.05,
            "validation_mean_excess_cumulative_return": 0.04,
            "validation_feasible_branch_rate": 1.00,
        },
        {
            "validation_annualized_return": 0.08,
            "validation_mean_excess_cumulative_return": 0.07,
            "validation_feasible_branch_rate": 1.00,
        },
        {
            "validation_annualized_return": 0.30,
            "validation_mean_excess_cumulative_return": 0.25,
            "validation_feasible_branch_rate": 0.95,
        },
    ]

    score, summary = trainer._best_feasible_from_metric_rows(rows)

    assert score == pytest.approx(0.07)
    assert summary is not None
    assert summary["feasible_branch_rate"] == pytest.approx(1.0)
    assert summary["mean_excess_cumulative_return"] == pytest.approx(0.07)

def test_v26_rcpo_smoke_uses_one_lambda_update_and_split_validation_log(
    tmp_path: Path,
) -> None:
    config = load_config(
        "configs/v2.6_experiment2_8assets/simplex_rcpo_gaussian.yaml"
    )
    config.experiment.output_root = str(tmp_path)
    config.experiment.run_name = "tiny_v26"
    config.market.lookback = 2
    config.market.train_market_count = 2
    config.market.train_steps = 40
    config.market.validation_steps = 8
    config.market.test_steps = 8
    config.environment.episode_length = 8
    config.optimization.total_updates = 1
    config.optimization.rollout_steps = 32
    config.optimization.epochs = 2
    config.optimization.minibatch_size = 16
    config.evaluation.validation_branch_count = 2
    config.evaluation.test_branch_count = 2
    config.evaluation.validation_interval_updates = 1

    run_dir = run_experiment(
        config,
        algo="rcpo",
        disable_artifacts=True,
    )[0]
    metric = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    validation = json.loads(
        (run_dir / "validation_metrics.jsonl").read_text().splitlines()[0]
    )

    assert metric["lambda_update_count"] == 1
    assert metric["optimizer_steps_attempted"] >= metric["optimizer_steps_completed"]
    assert metric["metrics_schema_version"] == 2
    assert not any(key.startswith("validation_") for key in metric)
    assert validation["validation_branches"] == 2
    assert "validation_feasible_branch_rate" in validation
    assert (run_dir / "checkpoint_best_return.pt").exists()
    assert (run_dir / "checkpoint_best_feasible.pt").exists()
    assert not (run_dir / "checkpoint_best.pt").exists()
    best_return = torch.load(
        run_dir / "checkpoint_best_return.pt", map_location="cpu"
    )
    best_feasible = torch.load(
        run_dir / "checkpoint_best_feasible.pt", map_location="cpu"
    )
    assert best_return["checkpoint_selection"] == "maximum_validation_return"
    assert best_feasible["checkpoint_selection"] == (
        "maximum_feasible_branch_rate_then_validation_return"
    )
    checkpoint = torch.load(
        run_dir / "checkpoint_last.pt", map_location="cpu"
    )
    assert checkpoint["observation_schema_version"] == 2
    assert checkpoint["constraint_semantics_version"] == "relative_current_drawdown_v1"




