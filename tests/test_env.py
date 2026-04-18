from __future__ import annotations

import numpy as np

from rcpo_portfolio.config import EnvironmentConfig, MarketConfig
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.market import MarketSlice


def build_env(**overrides) -> PortfolioEnv:
    returns = np.array(
        [
            [0.01, -0.02],
            [0.02, 0.01],
            [-0.01, 0.03],
            [0.01, 0.01],
            [-0.02, -0.01],
            [0.00, 0.02],
            [0.01, 0.01],
        ],
        dtype=np.float32,
    )
    regimes = np.zeros(len(returns), dtype=np.int64)
    market = MarketSlice(risky_returns=returns, regimes=regimes)
    market_config = MarketConfig(num_risky_assets=2, lookback=2, train_steps=5, test_steps=5)
    env_kwargs = {
        "episode_length": 3,
        "transaction_cost_bps": 10.0,
        "turnover_cap": 2.0,
        "group_a_indices": [0],
        "group_b_indices": [1],
        "active_constraint_preset": "c2",
        "constraint_presets": {
            "c1": {"group_a_min_weight": 0.20, "group_b_max_weight": 0.70},
            "c2": {"group_a_min_weight": 0.25, "group_b_max_weight": 0.60},
            "c3": {"group_a_min_weight": 0.30, "group_b_max_weight": 0.50},
        },
        "downside_cost_weight": 1.0,
        "group_a_min_cost_weight": 0.35,
        "group_b_max_cost_weight": 0.35,
    }
    env_kwargs.update(overrides)
    env_config = EnvironmentConfig(**env_kwargs)
    return PortfolioEnv(env_config, market, market_config, seed=7)


def test_action_projection_produces_long_only_weights() -> None:
    env = build_env()
    weights = env._weights_from_action(np.array([1.0, -1.0, 0.5], dtype=np.float32))
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert weights[0] >= 0.0


def test_reward_and_group_diagnostics_match_hand_calculation() -> None:
    env = build_env()
    env.reset(options={"start_index": 2})
    target_weights = np.array([0.2, 0.1, 0.7], dtype=np.float32)
    action = np.log(target_weights).astype(np.float32)
    _, reward, terminated, _, info = env.step(action)

    expected_raw_return = float(np.dot(target_weights[1:], np.array([-0.01, 0.03], dtype=np.float32)))
    expected_turnover = float(np.sum(np.abs(target_weights - np.array([1.0, 0.0, 0.0], dtype=np.float32))))
    expected_transaction_cost = 0.001 * expected_turnover
    expected_net = expected_raw_return - expected_transaction_cost
    expected_downside_cost = 0.0
    expected_group_a_violation = ((0.25 - 0.1) / 0.25) ** 2
    expected_group_b_violation = ((0.7 - 0.6) / 0.6) ** 2

    assert not terminated
    assert np.isclose(info["group_a_weight"], 0.1)
    assert np.isclose(info["group_b_weight"], 0.7)
    assert np.isclose(info["raw_return"], expected_raw_return)
    assert np.isclose(info["transaction_cost"], expected_transaction_cost)
    assert np.isclose(info["net_return"], expected_net)
    assert np.isclose(reward, np.log1p(expected_net))
    assert np.isclose(info["downside_cost"], expected_downside_cost)
    assert np.isclose(info["group_a_min_violation_cost"], expected_group_a_violation)
    assert np.isclose(info["group_b_max_violation_cost"], expected_group_b_violation)
    assert np.isclose(info["constraint_cost"], 0.0)
    assert info["constraint_mode"] == "downside"


def test_downside_constraint_cost_ignores_group_diagnostics() -> None:
    env = build_env()
    env.reset(options={"start_index": 2})
    target_weights = np.array([0.2, 0.7, 0.1], dtype=np.float32)
    action = np.log(target_weights).astype(np.float32)
    _, _, _, _, info = env.step(action)

    expected_raw_return = float(np.dot(target_weights[1:], np.array([-0.01, 0.03], dtype=np.float32)))
    expected_turnover = float(np.sum(np.abs(target_weights - np.array([1.0, 0.0, 0.0], dtype=np.float32))))
    expected_net = expected_raw_return - 0.001 * expected_turnover
    expected_downside_cost = max(0.0, -expected_net) ** 2

    assert np.isclose(info["net_return"], expected_net)
    assert np.isclose(info["downside_cost"], expected_downside_cost)
    assert np.isclose(info["normalized_downside_cost"], expected_downside_cost / 1e-4)
    assert np.isclose(info["constraint_cost"], info["normalized_downside_cost"])


def test_sortino_constraint_cost_uses_target_violation_after_warmup() -> None:
    env = build_env(
        constraint_mode="sortino",
        transaction_cost_bps=0.0,
        sortino_target=1.0,
        sortino_window=3,
        sortino_min_periods=2,
        sortino_cost_scale=1.0,
    )
    env.reset(options={"start_index": 2})
    target_weights = np.array([1e-6, 0.999998, 1e-6], dtype=np.float32)
    action = np.log(target_weights).astype(np.float32)

    _, _, _, _, first_info = env.step(action)
    _, _, _, _, second_info = env.step(action)
    returns = np.array([first_info["net_return"], second_info["net_return"]], dtype=np.float32)
    downside_deviation = np.sqrt(np.mean(np.square(np.minimum(returns, 0.0))))
    expected_sortino = np.sqrt(252.0) * float(np.mean(returns)) / float(downside_deviation)
    expected_cost = max(0.0, 1.0 - expected_sortino) ** 2

    assert first_info["sortino_violation_cost"] == 0.0
    assert first_info["constraint_cost"] == 0.0
    assert np.isclose(second_info["sortino_ratio"], expected_sortino)
    assert np.isclose(second_info["sortino_violation_cost"], expected_cost)
    assert np.isclose(second_info["constraint_cost"], expected_cost)


def test_constraint_preset_resolution_changes_bounds() -> None:
    env = build_env(active_constraint_preset="c3")
    resolved = env.resolved_constraint_preset()
    assert resolved["preset_name"] == "c3"
    assert np.isclose(resolved["group_a_min_weight"], 0.30)
    assert np.isclose(resolved["group_b_max_weight"], 0.50)


def test_reset_is_deterministic_when_start_index_is_fixed() -> None:
    env = build_env()
    first_obs, first_info = env.reset(options={"start_index": 2})
    second_obs, second_info = env.reset(options={"start_index": 2})
    np.testing.assert_allclose(first_obs, second_obs)
    assert first_info["start_index"] == second_info["start_index"] == 2
