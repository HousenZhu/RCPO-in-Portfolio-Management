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
    market_config = MarketConfig(
        num_risky_assets=2,
        lookback=2,
        train_steps=5,
        validation_steps=5,
        test_steps=5,
    )
    env_kwargs = {
        "episode_length": 3,
        "transaction_cost_bps": 10.0,
        "turnover_cap": 2.0,
        "constraint_mode": "max_drawdown",
        "drawdown_budget_floor": 0.02,
        "benchmark_drawdown_margin": 0.90,
        "drawdown_cost_scale": 0.01,
        "diversification_beta": 0.0,
        "group_a_indices": [0],
        "group_b_indices": [1],
        "active_constraint_preset": "c2",
        "constraint_presets": {
            "c1": {"group_a_min_weight": 0.20, "group_b_max_weight": 0.70},
            "c2": {"group_a_min_weight": 0.25, "group_b_max_weight": 0.60},
            "c3": {"group_a_min_weight": 0.30, "group_b_max_weight": 0.50},
        },
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


def test_reset_initializes_drawdown_tracking() -> None:
    env = build_env()
    obs, info = env.reset(options={"start_index": 2})

    assert obs.shape == env.observation_space.shape
    assert info["start_index"] == 2
    assert env.state is not None
    assert env.state.portfolio_value == 1.0
    assert env.state.running_peak_value == 1.0
    assert env.state.current_drawdown == 0.0
    assert env.state.max_drawdown == 0.0
    assert env.state.benchmark_portfolio_value == 1.0
    assert env.state.benchmark_running_peak_value == 1.0
    assert env.state.benchmark_current_drawdown == 0.0
    assert env.state.benchmark_max_drawdown == 0.0
    assert env.state.benchmark_has_rebalanced is False


def test_positive_return_updates_peak_and_keeps_drawdown_zero() -> None:
    env = build_env()
    env.reset(options={"start_index": 2})
    target_weights = np.array([0.2, 0.1, 0.7], dtype=np.float32)
    action = np.log(target_weights).astype(np.float32)

    _, reward, terminated, _, info = env.step(action)

    expected_raw_return = float(
        np.dot(target_weights[1:], np.array([-0.01, 0.03], dtype=np.float32))
    )
    expected_turnover = float(
        np.sum(np.abs(target_weights - np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    )
    expected_transaction_cost = 0.001 * expected_turnover
    expected_net = expected_raw_return - expected_transaction_cost
    expected_portfolio_value = 1.0 * (1.0 + expected_net)
    expected_benchmark_raw_return = float(
        np.dot(
            np.full(2, 1.0 / 3.0, dtype=np.float32),
            np.array([-0.01, 0.03], dtype=np.float32),
        )
    )
    expected_benchmark_turnover = 4.0 / 3.0
    expected_benchmark_transaction_cost = 0.001 * expected_benchmark_turnover
    expected_benchmark_net = (
        expected_benchmark_raw_return - expected_benchmark_transaction_cost
    )
    expected_benchmark_value = 1.0 * (1.0 + expected_benchmark_net)

    assert not terminated
    assert np.isclose(info["raw_return"], expected_raw_return)
    assert np.isclose(info["transaction_cost"], expected_transaction_cost)
    assert np.isclose(info["net_return"], expected_net)
    assert np.isclose(reward, np.log1p(expected_net))
    assert np.isclose(info["portfolio_value"], expected_portfolio_value)
    assert np.isclose(info["running_peak_value"], expected_portfolio_value)
    assert np.isclose(info["current_drawdown"], 0.0)
    assert np.isclose(info["max_drawdown"], 0.0)
    assert np.isclose(info["benchmark_raw_return"], expected_benchmark_raw_return)
    assert np.isclose(info["benchmark_turnover"], expected_benchmark_turnover)
    assert np.isclose(
        info["benchmark_transaction_cost"],
        expected_benchmark_transaction_cost,
    )
    assert np.isclose(info["benchmark_net_return"], expected_benchmark_net)
    assert np.isclose(info["benchmark_portfolio_value"], expected_benchmark_value)
    assert np.isclose(info["benchmark_running_peak_value"], expected_benchmark_value)
    assert np.isclose(info["benchmark_current_drawdown"], 0.0)
    assert np.isclose(info["benchmark_max_drawdown"], 0.0)
    assert np.isclose(info["effective_drawdown_budget"], 0.02)
    assert np.isclose(info["drawdown_gap"], -0.02)
    assert np.isclose(info["drawdown_violation"], 0.0)
    assert np.isclose(info["drawdown_constraint_cost"], 0.0)
    assert np.isclose(info["constraint_cost"], 0.0)
    assert info["constraint_mode"] == "max_drawdown"
    assert "drawdown_budget" not in info
    assert "downside_cost" not in info
    assert "normalized_downside_cost" not in info
    assert "sortino_ratio" not in info
    assert "sortino_violation_cost" not in info


def test_first_benchmark_step_applies_initial_rebalance_cost() -> None:
    env = build_env(drawdown_budget_floor=0.001)
    env.reset(options={"start_index": 4})

    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    expected_raw_return = float((-0.02 - 0.01) / 3.0)
    expected_turnover = 4.0 / 3.0
    expected_transaction_cost = 0.001 * expected_turnover
    expected_net_return = expected_raw_return - expected_transaction_cost
    expected_max_drawdown = -expected_net_return
    expected_budget = max(0.001, 0.90 * expected_max_drawdown)

    assert np.isclose(info["benchmark_raw_return"], expected_raw_return)
    assert np.isclose(info["benchmark_turnover"], expected_turnover)
    assert np.isclose(info["benchmark_transaction_cost"], expected_transaction_cost)
    assert np.isclose(info["benchmark_net_return"], expected_net_return)
    assert np.isclose(info["benchmark_current_drawdown"], expected_max_drawdown)
    assert np.isclose(info["benchmark_max_drawdown"], expected_max_drawdown)
    assert np.isclose(info["effective_drawdown_budget"], expected_budget)


def test_later_benchmark_steps_have_zero_turnover() -> None:
    env = build_env(drawdown_budget_floor=0.001)
    env.reset(options={"start_index": 2})
    equal_weight_action = np.zeros(3, dtype=np.float32)

    _, _, _, _, first_info = env.step(equal_weight_action)
    _, _, _, _, second_info = env.step(equal_weight_action)

    assert first_info["benchmark_turnover"] > 0.0
    assert np.isclose(second_info["benchmark_turnover"], 0.0)
    assert np.isclose(second_info["benchmark_transaction_cost"], 0.0)


def test_constraint_cost_uses_online_benchmark_relative_budget() -> None:
    env = build_env(
        drawdown_budget_floor=0.001,
        benchmark_drawdown_margin=0.90,
        drawdown_cost_scale=0.01,
    )
    env.reset(options={"start_index": 4})
    target_weights = np.array([0.02, 0.97, 0.01], dtype=np.float32)

    _, _, _, _, info = env.step(np.log(target_weights).astype(np.float32))

    expected_benchmark_raw = float((-0.02 - 0.01) / 3.0)
    expected_benchmark_turnover = 4.0 / 3.0
    expected_benchmark_transaction_cost = 0.001 * expected_benchmark_turnover
    expected_benchmark_net = expected_benchmark_raw - expected_benchmark_transaction_cost
    expected_benchmark_max_drawdown = -expected_benchmark_net
    expected_budget = max(0.001, 0.90 * expected_benchmark_max_drawdown)

    expected_raw = float(
        np.dot(target_weights[1:], np.array([-0.02, -0.01], dtype=np.float32))
    )
    expected_turnover = float(
        np.sum(np.abs(target_weights - np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    )
    expected_transaction_cost = 0.001 * expected_turnover
    expected_net = expected_raw - expected_transaction_cost
    expected_agent_max_drawdown = -expected_net
    expected_gap = expected_agent_max_drawdown - expected_budget
    expected_violation = max(0.0, expected_gap)
    expected_cost = expected_violation**2 / 0.01

    assert np.isclose(info["benchmark_max_drawdown"], expected_benchmark_max_drawdown)
    assert np.isclose(info["effective_drawdown_budget"], expected_budget)
    assert np.isclose(info["max_drawdown"], expected_agent_max_drawdown)
    assert np.isclose(info["drawdown_gap"], expected_gap)
    assert np.isclose(info["drawdown_violation"], expected_violation)
    assert np.isclose(info["drawdown_constraint_cost"], expected_cost)
    assert np.isclose(info["constraint_cost"], expected_cost)


def test_equal_weight_can_have_positive_constraint_cost_under_margin_budget() -> None:
    env = build_env(drawdown_budget_floor=0.0, benchmark_drawdown_margin=0.90)
    env.reset(options={"start_index": 4})

    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    assert np.isclose(info["weights"].sum(), 1.0)
    assert info["benchmark_max_drawdown"] > 0.0
    assert info["drawdown_gap"] > 0.0
    assert info["constraint_cost"] > 0.0


def test_equal_weights_have_zero_excess_concentration_cost() -> None:
    env = build_env()
    env.reset(options={"start_index": 2})
    action = np.zeros(3, dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert np.isclose(info["weights"].sum(), 1.0)
    assert np.allclose(info["weights"], np.full(3, 1.0 / 3.0))
    assert np.isclose(info["concentration"], 1.0 / 3.0)
    assert np.isclose(info["excess_concentration_cost"], 0.0)
    assert np.isclose(info["diversification_cost"], 0.0)


def test_concentrated_weights_have_positive_diversification_diagnostics() -> None:
    env = build_env(diversification_beta=0.10)
    env.reset(options={"start_index": 2})
    target_weights = np.array([0.02, 0.97, 0.01], dtype=np.float32)
    _, _, _, _, info = env.step(np.log(target_weights).astype(np.float32))

    assert info["concentration"] > 1.0 / 3.0
    assert info["excess_concentration_cost"] > 0.0
    assert info["diversification_cost"] > 0.0
    assert np.isclose(info["constraint_cost"], info["drawdown_constraint_cost"])


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
