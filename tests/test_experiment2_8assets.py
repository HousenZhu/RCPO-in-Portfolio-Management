from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rcpo_portfolio.config import load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.market import _regime_model, generate_market_split
from rcpo_portfolio.models import ActorCritic


EXPERIMENT_CONFIGS = [
    "configs/experiment2_8assets/simplex_ppo_gaussian.yaml",
    "configs/experiment2_8assets/simplex_rcpo_gaussian.yaml",
    "configs/experiment2_8assets/simplex_ppo_dirichlet.yaml",
    "configs/experiment2_8assets/simplex_rcpo_dirichlet.yaml",
    "configs/experiment2_8assets/rcpo_allocation_penalty.yaml",
    "configs/experiment2_8assets/rcpo_allocation_drawdown_penalty.yaml",
]


def test_config_inheritance_resolves_relative_base_file(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "market:\n  num_risky_assets: 8\nnetwork:\n  hidden_sizes: [192, 128]\n",
        encoding="utf-8",
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        "extends: base.yaml\nmarket:\n  lookback: 24\n",
        encoding="utf-8",
    )

    config = load_config(child_path)

    assert config.market.num_risky_assets == 8
    assert config.market.lookback == 24
    assert config.network.hidden_sizes == [192, 128]


def test_experiment2_market_has_rotating_winners_and_weak_overlap() -> None:
    config = load_config("configs/experiment2_8assets/base.yaml")
    drifts, _ = _regime_model(config.market)
    low_vol = drifts[0]
    high_vol = drifts[1]

    assert len(low_vol) == 8
    assert np.argsort(low_vol)[-2:].tolist() == [1, 0]
    assert np.argsort(high_vol)[-3:].tolist() == [5, 6, 7]

    overlap_risky_index = 3  # Full-portfolio asset 4.
    assert low_vol[overlap_risky_index] < low_vol[0]
    assert low_vol[overlap_risky_index] < low_vol[1]
    assert high_vol[overlap_risky_index] < high_vol[5]
    assert high_vol[overlap_risky_index] < high_vol[6]
    assert high_vol[overlap_risky_index] < high_vol[7]

    group_1 = set(config.environment.allocation_constraint_1_indices)
    group_2 = set(config.environment.allocation_constraint_2_indices)
    assert group_1 == {1, 4, 6}
    assert group_2 == {2, 4, 7}
    assert group_1 & group_2 == {4}
    active = config.environment.constraint_presets["c3"]
    assert active["allocation_constraint_1_min_weight"] == 0.50
    assert active["allocation_constraint_2_min_weight"] == 0.40
    assert (
        active["allocation_constraint_1_min_weight"]
        + active["allocation_constraint_2_min_weight"] < 1.0
    )


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS)
def test_experiment2_configs_build_expected_environment_and_model(
    config_path: str,
) -> None:
    config = load_config(config_path)
    sync_rcpo_constraint_settings(config)
    config.environment.episode_length = 16
    market = generate_market_split(config.market, steps=48, seed=9)
    env = PortfolioEnv(config.environment, market, config.market, seed=9)

    assert config.market.num_risky_assets == 8
    assert config.market.lookback == 20
    assert config.network.hidden_sizes == [192, 128]
    assert env.observation_space.shape == (191,)

    if config.environment.action_mode == "simplex_decomposition":
        assert env.simplex_branch_sizes() == [1, 3, 3, 9]
        assert env.action_space.shape == (16,)
        weights = env._weights_from_action(env.neutral_action())
        preset = env.resolved_constraint_preset()
        assert weights[config.environment.allocation_constraint_1_indices].sum() >= (
            preset["allocation_constraint_1_min_weight"] - 1e-6
        )
        assert weights[config.environment.allocation_constraint_2_indices].sum() >= (
            preset["allocation_constraint_2_min_weight"] - 1e-6
        )
        assert env.simplex_branch_train_mask() == [False, True, True, True]
        assert env._simplex_decomposition is not None
        if config.environment.simplex_action_format == "branch_weights":
            decomposition = env._simplex_decomposition.map_branch_weights(
                env.neutral_action()
            )
        else:
            decomposition = env._simplex_decomposition.map_logits(env.neutral_action())
        assert decomposition.diagnostics["simplex_z1"] == 0.0
        assert weights[4] < 0.30
    else:
        assert env.action_space.shape == (9,)

    model = ActorCritic(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config.network,
        branch_sizes=env.simplex_branch_sizes(),
        branch_train_mask=env.simplex_branch_train_mask(),
    )
    output = model.get_policy_output(
        torch.zeros((2, env.observation_space.shape[0])),
        deterministic=True,
    )

    assert output.action.shape == (2, env.action_space.shape[0])
    assert torch.isfinite(output.action).all()
    assert torch.isfinite(output.log_prob).all()


def test_experiment2_soft_rcpo_initial_penalty_is_scaled_but_nonzero() -> None:
    config = load_config(
        "configs/experiment2_8assets/rcpo_allocation_penalty.yaml"
    )
    sync_rcpo_constraint_settings(config)
    config.environment.episode_length = 16
    market = generate_market_split(config.market, steps=48, seed=17)
    env = PortfolioEnv(config.environment, market, config.market, seed=17)
    env.reset()

    baseline_action = env.constrained_neutral_action()
    baseline_weights = env._weights_from_action(baseline_action)
    np.testing.assert_allclose(baseline_weights, env.benchmark_weights(), atol=1e-7)
    assert baseline_weights[
        config.environment.allocation_constraint_1_indices
    ].sum() >= 0.50 - 1e-6
    assert baseline_weights[
        config.environment.allocation_constraint_2_indices
    ].sum() >= 0.40 - 1e-6

    _, _, _, _, info = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
    uniform_group_weight = 3.0 / 9.0
    expected_raw_cost = (
        ((0.50 - uniform_group_weight) / 0.50) ** 2
        + ((0.40 - uniform_group_weight) / 0.40) ** 2
    )
    expected_scaled_cost = (
        expected_raw_cost / config.environment.allocation_constraint_cost_scale
    )

    assert info["allocation_constraint_raw_cost"] == pytest.approx(expected_raw_cost)
    assert info["allocation_constraint_cost"] == pytest.approx(expected_scaled_cost)
    assert 0.0 < info["allocation_constraint_cost"] < 0.01
