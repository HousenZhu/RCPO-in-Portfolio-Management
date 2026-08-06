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
    config.environment.action_mode = "simplex_decomposition"
    config.network.policy_architecture = "simplex_branch_gaussian"
    config.environment.drawdown_budget_floor = 0.02
    config.environment.drawdown_benchmark_mode = "constrained_neutral"
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
    config.reward_correction.gdrc_candidate_bins = [48, 64]
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
    assert (rcpo_run / "checkpoint_best_return.pt").exists()
    assert (rcpo_drc_run / "checkpoint_best_return.pt").exists()
    assert (rcpo_gdrc_run / "checkpoint_best_return.pt").exists()
    assert (ppo_run / "checkpoint_last.pt").exists()
    assert (ppo_run / "checkpoint_best_return.pt").exists()
    assert (ppo_drc_run / "checkpoint_best_return.pt").exists()
    assert (ppo_gdrc_run / "checkpoint_best_return.pt").exists()
    assert (rcpo_run / "checkpoint_best_feasible.pt").exists()
    assert (ppo_run / "checkpoint_best_feasible.pt").exists()
    assert not (rcpo_run / "checkpoint_best.pt").exists()
    assert not (ppo_run / "checkpoint_best.pt").exists()

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
    assert "average_allocation_constraint_1_violation_cost" in payload
    assert "average_allocation_constraint_1_weight" in payload
    assert "average_simplex_z1" in payload

    with (rcpo_run / "config_snapshot.yaml").open("r", encoding="utf-8") as handle:
        snapshot = handle.read()
    assert "action_mode: simplex_decomposition" in snapshot
    assert "simplex_action_format: branch_logits" in snapshot
    assert "policy_architecture: simplex_branch_gaussian" in snapshot
    assert "active_constraint_preset: c2" in snapshot
    assert "constraint_mode: max_drawdown" in snapshot
    assert "drawdown_budget_floor: 0.02" in snapshot
    assert "drawdown_benchmark_mode: constrained_neutral" in snapshot
    assert "benchmark_drawdown_margin: 0.9" in snapshot
    assert "drawdown_cost_scale: 0.01" in snapshot
    assert "alpha_budget_ratio: 0.05" in snapshot
    assert "lambda_lr_up: 0.015" in snapshot
    assert "lambda_lr_down: 0.03" in snapshot
    assert "reward_noise:" in snapshot
    assert "enabled: false" in snapshot
    assert "runtime:" in snapshot
    assert "device: cpu" in snapshot
    assert "resolved_allocation_constraint_1_min_weight: 0.4" in snapshot

    with (ppo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        ppo_metric = json.loads(handle.readline())
    assert "approx_kl" in ppo_metric
    assert "clip_fraction" in ppo_metric
    assert "learning_rate" in ppo_metric
    assert "validation_annualized_return" in ppo_metric
    assert ppo_metric["learning_rate"] == config.ppo.learning_rate
    assert ppo_metric["reward_noise_enabled"] == 0
    assert ppo_metric["batch_reward_noise_std"] == 0.0

    with (rcpo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rcpo_metric = json.loads(handle.readline())
    assert rcpo_metric["learning_rate"] == config.optimization.learning_rate
    assert rcpo_metric["constraint_mode"] == "max_drawdown"
    assert rcpo_metric["action_mode"] == "simplex_decomposition"
    assert rcpo_metric["simplex_action_format"] == "branch_logits"
    assert rcpo_metric["policy_architecture"] == "simplex_branch_gaussian"
    assert rcpo_metric["drawdown_benchmark_mode"] == "constrained_neutral"
    assert rcpo_metric["alpha_mode"] == "budget_ratio"
    assert rcpo_metric["alpha_budget_ratio"] == config.rcpo.alpha_budget_ratio
    assert rcpo_metric["lambda_lr_up"] == config.rcpo.lambda_lr_up
    assert rcpo_metric["lambda_lr_down"] == config.rcpo.lambda_lr_down
    assert "lambda_gap" in rcpo_metric
    assert rcpo_metric["alpha"] == rcpo_metric["batch_alpha_target_mean"]
    assert rcpo_metric["alpha"] > 0.0
    assert "validation_turnover" in rcpo_metric
    assert "validation_mean_excess_cumulative_return" in rcpo_metric
    assert "validation_mean_relative_wealth_vs_constrained_neutral" in rcpo_metric
    assert "episode_relative_wealth_vs_baseline_mean" in rcpo_metric
    assert "validation_win_rate_vs_equal_weight" in rcpo_metric
    assert "validation_max_drawdown" in rcpo_metric
    assert "validation_benchmark_max_drawdown" in rcpo_metric
    assert rcpo_metric["validation_drawdown_benchmark_mode"] == "constrained_neutral"
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
    assert "batch_true_reward_mean" in rcpo_metric
    assert "batch_observed_reward_mean" in rcpo_metric
    assert "batch_reward_noise_mean" in rcpo_metric
    assert "batch_reward_noise_std" in rcpo_metric
    assert "batch_allocation_constraint_1_violation_cost_mean" in rcpo_metric
    assert "batch_allocation_constraint_2_violation_cost_mean" in rcpo_metric
    assert "batch_allocation_constraint_1_weight_mean" in rcpo_metric
    assert "batch_allocation_constraint_2_weight_mean" in rcpo_metric
    assert "batch_simplex_z1_mean" in rcpo_metric
    assert "validation_allocation_constraint_1_weight" in rcpo_metric
    assert "validation_allocation_constraint_2_weight" in rcpo_metric
    assert "validation_allocation_constraint_1_violation_cost" in rcpo_metric
    assert "validation_allocation_constraint_2_violation_cost" in rcpo_metric

    assert rcpo_metric["reward_noise_enabled"] == 0
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
    assert rcpo_gdrc_metric["gdrc_selected_bins"] in {48, 64}
    assert rcpo_gdrc_metric["gdrc_candidate_bins"] == [48, 64]
    assert rcpo_gdrc_metric["reward_correction_coef"] == pytest.approx(0.50)
    assert rcpo_gdrc_metric["reward_correction_delta_clip"] == pytest.approx(0.0015)
    assert "reward_correction_raw_delta_abs_mean" in rcpo_gdrc_metric
    assert "reward_correction_effective_delta_abs_mean" in rcpo_gdrc_metric
    assert (rcpo_gdrc_run / "evaluation" / "gdrc_selected_bins.png").exists()

    original_metric_count = sum(1 for _ in (ppo_run / "metrics.jsonl").open("r", encoding="utf-8"))
    config.reward_correction.mode = "none"
    config.ppo.total_updates = 1
    resumed_run = resume_experiment(config, algo="ppo_unconstrained", run_dir=ppo_run)
    resumed_metric_count = sum(1 for _ in (resumed_run / "metrics.jsonl").open("r", encoding="utf-8"))
    assert resumed_metric_count == original_metric_count + 1


def test_short_rcpo_allocation_penalty_run(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.experiment.run_name = "tiny_rcpo_allocation"
    config.environment.action_mode = "softmax"
    config.environment.initial_portfolio_mode = "constrained_neutral"
    config.environment.constraint_mode = "allocation"
    config.environment.allocation_constraint_cost_scale = 20.0
    config.network.policy_architecture = "flat_gaussian"
    config.network.branch_credit_mode = "global"
    config.rcpo.constraint_mode = "allocation"
    config.rcpo.alpha = 0.0005
    config.rcpo.lambda_lr_up = 0.0015
    config.rcpo.lambda_lr_down = 0.03
    config.reward_correction.mode = "none"

    run_dir = run_experiment(config, algo="rcpo")[0]
    metric = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    snapshot = (run_dir / "config_snapshot.yaml").read_text()

    assert "constraint_mode: allocation" in snapshot
    assert "allocation_constraint_cost_scale: 20.0" in snapshot
    assert metric["constraint_mode"] == "allocation"
    assert "batch_allocation_constraint_cost_mean" in metric
    assert "batch_allocation_constraint_raw_cost_mean" in metric
    assert "batch_lambda_cost_advantage_ratio" in metric
    assert "lambda_cost_to_reward_ratio" in metric
    assert metric["batch_constraint_cost_mean"] == pytest.approx(
        metric["batch_allocation_constraint_cost_mean"]
    )


def test_short_rcpo_allocation_drawdown_run_saves_feasible_best(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.experiment.run_name = "tiny_rcpo_allocation_drawdown"
    config.environment.action_mode = "softmax"
    config.environment.initial_portfolio_mode = "constrained_neutral"
    config.environment.constraint_mode = "allocation_drawdown"
    config.environment.allocation_constraint_cost_scale = 20.0
    config.environment.combined_drawdown_cost_weight = 0.25
    config.network.policy_architecture = "flat_gaussian"
    config.network.branch_credit_mode = "global"
    config.rcpo.constraint_mode = "allocation_drawdown"
    config.rcpo.alpha = 1.0
    config.rcpo.lambda_lr_up = 0.0005
    config.rcpo.lambda_lr_down = 0.03
    config.reward_correction.mode = "none"

    run_dir = run_experiment(config, algo="rcpo")[0]
    metric = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    summary = json.loads((run_dir / "training_summary.json").read_text())

    assert metric["constraint_mode"] == "allocation_drawdown"
    assert metric["validation_constraint_feasible"] == 1
    assert metric["validation_feasible_best"] == 1
    assert metric["batch_constraint_cost_mean"] == pytest.approx(
        metric["batch_allocation_drawdown_constraint_cost_mean"]
    )
    assert (run_dir / "checkpoint_best_feasible.pt").exists()
    assert (run_dir / "evaluation_best_feasible" / "summary_validation.json").exists()
    assert summary["best_feasible_validation"] is not None
    assert summary["evaluation_best_feasible"]


def test_short_noisy_reward_training_runs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config.reward_noise.enabled = True
    config.reward_noise.std = 0.003
    config.reward_correction.mode = "none"
    config.experiment.run_name = "tiny_noisy_ppo"
    ppo_run = run_experiment(config, algo="ppo_unconstrained")[0]

    config.reward_correction.mode = "gdrc"
    config.experiment.run_name = "tiny_noisy_ppo_gdrc"
    ppo_gdrc_run = run_experiment(config, algo="ppo_unconstrained")[0]

    config.reward_correction.mode = "none"
    config.experiment.run_name = "tiny_noisy_rcpo"
    rcpo_run = run_experiment(config, algo="rcpo")[0]

    with (ppo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        ppo_metric = json.loads(handle.readline())
    assert ppo_metric["reward_noise_enabled"] == 1
    assert ppo_metric["reward_noise_std"] == pytest.approx(0.003)
    assert ppo_metric["batch_reward_noise_std"] > 0.0
    assert ppo_metric["batch_true_reward_mean"] != ppo_metric["batch_observed_reward_mean"]

    with (ppo_gdrc_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        ppo_gdrc_metric = json.loads(handle.readline())
    assert ppo_gdrc_metric["reward_noise_enabled"] == 1
    assert ppo_gdrc_metric["reward_correction_mode"] == "gdrc"
    assert ppo_gdrc_metric["gdrc_selected_bins"] in {48, 64}
    assert "reward_correction_delta_abs_mean" in ppo_gdrc_metric
    assert "reward_correction_raw_delta_abs_mean" in ppo_gdrc_metric

    with (rcpo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rcpo_metric = json.loads(handle.readline())
    assert rcpo_metric["reward_noise_enabled"] == 1
    assert rcpo_metric["constraint_mode"] == "max_drawdown"
    assert "batch_drawdown_constraint_cost_mean" in rcpo_metric


def test_short_autoregressive_gaussian_policy_training_runs_for_ppo_and_rcpo(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path)
    config.reward_correction.mode = "none"
    config.network.policy_architecture = "simplex_autoregressive_gaussian"
    config.experiment.run_name = "tiny_autoregressive_gaussian_ppo"
    ppo_run = run_experiment(config, algo="ppo_unconstrained")[0]

    config.experiment.run_name = "tiny_autoregressive_gaussian_rcpo"
    rcpo_run = run_experiment(config, algo="rcpo")[0]

    with (ppo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        ppo_metric = json.loads(handle.readline())
    assert ppo_metric["policy_architecture"] == "simplex_autoregressive_gaussian"
    assert ppo_metric["simplex_action_format"] == "branch_logits"
    assert np.isfinite(ppo_metric["approx_kl"])

    with (rcpo_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rcpo_metric = json.loads(handle.readline())
    assert rcpo_metric["policy_architecture"] == "simplex_autoregressive_gaussian"
    assert rcpo_metric["simplex_action_format"] == "branch_logits"
    assert "validation_allocation_constraint_1_weight" in rcpo_metric
    assert (rcpo_run / "evaluation_best" / "summary_test.json").exists()


@pytest.mark.parametrize(
    "policy_architecture,expected_action_format",
    [
        ("simplex_autoregressive_gaussian", "branch_logits"),
        ("simplex_autoregressive_dirichlet", "branch_weights"),
    ],
)
@pytest.mark.parametrize("algo", ["ppo_unconstrained", "rcpo"])
def test_short_standalone_branch_credit_training_runs(
    tmp_path: Path,
    policy_architecture: str,
    expected_action_format: str,
    algo: str,
) -> None:
    config = tiny_config(tmp_path)
    config.network.policy_architecture = policy_architecture
    config.network.branch_credit_mode = "standalone"
    config.environment.initial_portfolio_mode = "constrained_neutral"
    config.reward_correction.mode = "none"
    config.reward_noise.enabled = False
    config.optimization.rollout_steps = 32
    config.optimization.minibatch_size = 16
    config.ppo.rollout_steps = 32
    config.ppo.minibatch_size = 16
    architecture_tag = "dirichlet" if policy_architecture.endswith("dirichlet") else "gaussian"
    algo_tag = "ppo" if algo == "ppo_unconstrained" else "rcpo"
    config.experiment.run_name = f"tiny_standalone_{architecture_tag}_{algo_tag}"

    run_dir = run_experiment(config, algo=algo)[0]

    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        metric = json.loads(handle.readline())
    assert metric["policy_architecture"] == policy_architecture
    assert metric["simplex_action_format"] == expected_action_format
    assert metric["branch_credit_mode"] == "standalone"
    assert metric["initial_portfolio_mode"] == "constrained_neutral"
    assert metric["optimizer_steps_completed"] > 0
    assert "trigger_minibatch_kl" in metric
    for branch_number in range(1, 5):
        assert f"approx_kl_branch_{branch_number}" in metric
        assert f"entropy_branch_{branch_number}" in metric
        assert f"batch_branch_{branch_number}_reward_mean" in metric
        assert f"batch_branch_{branch_number}_transaction_cost_mean" in metric
        assert f"batch_branch_{branch_number}_drawdown_cost_mean" in metric
        assert f"batch_branch_{branch_number}_reward_advantage_std" in metric
        assert f"batch_branch_{branch_number}_cost_advantage_std" in metric
        assert f"batch_branch_{branch_number}_z_mean" in metric

    checkpoint = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    assert checkpoint["branch_credit_mode"] == "standalone"
    assert checkpoint["initial_portfolio_mode"] == "constrained_neutral"
    assert checkpoint["policy_architecture"] == policy_architecture
    assert checkpoint["simplex_action_format"] == expected_action_format


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

    benchmark_mismatch_path = run_dir / "checkpoint_benchmark_mismatch.pt"
    payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    payload["drawdown_benchmark_mode"] = "true_equal_weight"
    torch.save(payload, benchmark_mismatch_path)

    with pytest.raises(ValueError, match="drawdown_benchmark_mode"):
        resume_experiment(
            config,
            algo="rcpo",
            run_dir=run_dir,
            checkpoint_name="checkpoint_benchmark_mismatch.pt",
        )

    mismatch_path = run_dir / "checkpoint_policy_mismatch.pt"
    payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    payload["policy_architecture"] = "flat_gaussian"
    torch.save(payload, mismatch_path)

    with pytest.raises(ValueError, match="policy_architecture"):
        resume_experiment(
            config,
            algo="rcpo",
            run_dir=run_dir,
            checkpoint_name="checkpoint_policy_mismatch.pt",
        )

    branch_credit_mismatch_path = run_dir / "checkpoint_branch_credit_mismatch.pt"
    payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    payload["branch_credit_mode"] = "standalone"
    torch.save(payload, branch_credit_mismatch_path)

    with pytest.raises(ValueError, match="branch_credit_mode"):
        resume_experiment(
            config,
            algo="rcpo",
            run_dir=run_dir,
            checkpoint_name="checkpoint_branch_credit_mismatch.pt",
        )

    initial_mode_mismatch_path = run_dir / "checkpoint_initial_mode_mismatch.pt"
    payload = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
    payload["initial_portfolio_mode"] = "constrained_neutral"
    torch.save(payload, initial_mode_mismatch_path)

    with pytest.raises(ValueError, match="initial_portfolio_mode"):
        resume_experiment(
            config,
            algo="rcpo",
            run_dir=run_dir,
            checkpoint_name="checkpoint_initial_mode_mismatch.pt",
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
