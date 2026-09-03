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
        "action_mode": "softmax",
        "constraint_mode": "max_drawdown",
        "drawdown_budget_floor": 0.02,
        "benchmark_drawdown_margin": 0.90,
        "drawdown_cost_scale": 0.01,
        "diversification_beta": 0.0,
        "allocation_constraint_1_indices": [0],
        "allocation_constraint_2_indices": [1],
        "active_constraint_preset": "c2",
        "constraint_presets": {
            "c1": {
                "allocation_constraint_1_min_weight": 0.20,
                "allocation_constraint_2_min_weight": 0.70,
            },
            "c2": {
                "allocation_constraint_1_min_weight": 0.25,
                "allocation_constraint_2_min_weight": 0.60,
            },
            "c3": {
                "allocation_constraint_1_min_weight": 0.30,
                "allocation_constraint_2_min_weight": 0.50,
            },
        },
    }
    env_kwargs.update(overrides)
    env_config = EnvironmentConfig(**env_kwargs)
    return PortfolioEnv(env_config, market, market_config, seed=7)


def build_project_simplex_env(**overrides) -> PortfolioEnv:
    returns = np.array(
        [
            [0.01, -0.02, 0.00, 0.01, -0.01],
            [0.02, 0.01, -0.01, 0.00, 0.02],
            [-0.01, 0.03, 0.01, -0.02, -0.01],
            [0.01, 0.01, 0.00, 0.02, 0.00],
            [-0.02, -0.01, 0.01, -0.01, -0.02],
            [0.00, 0.02, -0.01, 0.01, 0.01],
            [0.01, 0.01, 0.00, 0.00, 0.02],
        ],
        dtype=np.float32,
    )
    regimes = np.zeros(len(returns), dtype=np.int64)
    market = MarketSlice(risky_returns=returns, regimes=regimes)
    market_config = MarketConfig(
        num_risky_assets=5,
        lookback=2,
        train_steps=5,
        validation_steps=5,
        test_steps=5,
    )
    env_kwargs = {
        "episode_length": 3,
        "transaction_cost_bps": 10.0,
        "turnover_cap": 2.0,
        "action_mode": "simplex_decomposition",
        "simplex_action_format": "branch_logits",
        "constraint_mode": "max_drawdown",
        "drawdown_budget_floor": 0.02,
        "drawdown_benchmark_mode": "constrained_neutral",
        "benchmark_drawdown_margin": 0.90,
        "drawdown_cost_scale": 0.01,
        "diversification_beta": 0.0,
        "allocation_constraint_1_indices": [1, 2, 4],
        "allocation_constraint_2_indices": [0, 4, 5],
        "active_constraint_preset": "c3",
        "constraint_presets": {
            "c1": {
                "allocation_constraint_1_min_weight": 0.35,
                "allocation_constraint_2_min_weight": 0.35,
            },
            "c2": {
                "allocation_constraint_1_min_weight": 0.40,
                "allocation_constraint_2_min_weight": 0.40,
            },
            "c3": {
                "allocation_constraint_1_min_weight": 0.55,
                "allocation_constraint_2_min_weight": 0.55,
            },
        },
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


def test_constrained_neutral_reset_starts_without_initial_rebalance_cost() -> None:
    env = build_project_simplex_env(initial_portfolio_mode="constrained_neutral")
    neutral_action = env.neutral_action()
    expected_weights = env._weights_from_action(neutral_action)

    env.reset(options={"start_index": 2})
    assert env.state is not None
    np.testing.assert_allclose(env.state.weights, expected_weights)
    assert env.state.benchmark_has_rebalanced is True

    _, _, _, _, info = env.step(neutral_action)
    assert np.isclose(info["turnover"], 0.0)
    assert np.isclose(info["transaction_cost"], 0.0)
    assert np.isclose(info["benchmark_turnover"], 0.0)
    np.testing.assert_allclose(info["branch_turnovers"], np.zeros(4), atol=1e-8)


def test_counterfactual_neutral_replacement_is_online_and_zero_at_neutral_action() -> None:
    env = build_project_simplex_env(
        initial_portfolio_mode="constrained_neutral",
        counterfactual_branch_credit_enabled=True,
    )
    neutral_action = env.neutral_action()
    env.reset(options={"start_index": 2})

    context = env.counterfactual_critic_context()
    assert context.shape == (4, env.num_assets + 6)
    _, reward, _, _, info = env.step(neutral_action)

    active = np.asarray(env.simplex_branch_train_mask(), dtype=bool)
    np.testing.assert_allclose(info["branch_delta_rewards"][active], 0.0, atol=1e-8)
    np.testing.assert_allclose(info["branch_delta_costs"][active], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        info["counterfactual_branch_rewards"][active], reward, atol=1e-8
    )
    assert info["counterfactual_nonfinite_count"] == 0
    assert info["counterfactual_mapping_failure_count"] == 0


def test_counterfactual_branch_uses_own_turnover_and_stateful_path() -> None:
    env = build_project_simplex_env(
        initial_portfolio_mode="constrained_neutral",
        counterfactual_branch_credit_enabled=True,
        drawdown_budget_floor=0.001,
    )
    env.reset(options={"start_index": 4})
    action = env.neutral_action()
    branch_sizes = env.simplex_branch_sizes()
    branch_2_start = branch_sizes[0]
    action[branch_2_start : branch_2_start + branch_sizes[1]] = np.asarray(
        [4.0, -2.0, -2.0], dtype=np.float32
    )

    _, reward, _, _, first = env.step(action)
    assert first["counterfactual_weight_l1_distances"][1] > 0.0
    assert np.isclose(
        first["branch_delta_rewards"][1],
        reward - first["counterfactual_branch_rewards"][1],
    )
    first_counterfactual_weights = env.state.counterfactual_weights[1].copy()

    env.step(action)
    np.testing.assert_allclose(
        env.state.counterfactual_weights[1], first_counterfactual_weights, atol=1e-8
    )
    assert env.state.counterfactual_previous_turnovers[1] >= 0.0


def test_standalone_branch_reward_and_cost_use_same_market_step() -> None:
    env = build_project_simplex_env(
        initial_portfolio_mode="constrained_neutral",
        drawdown_budget_floor=0.001,
    )
    action = env.neutral_action()
    env.reset(options={"start_index": 4})
    assert env.state is not None
    initial_branches = tuple(weights.copy() for weights in env.state.branch_weights)
    current_returns = env.market.risky_returns[4]

    _, _, _, _, info = env.step(action)

    branch_train_mask = env.simplex_branch_train_mask()
    for branch_index, branch_weights in enumerate(initial_branches):
        if not branch_train_mask[branch_index]:
            assert np.isclose(info["branch_rewards"][branch_index], 0.0)
            assert np.isclose(info["branch_transaction_costs"][branch_index], 0.0)
            assert np.isclose(info["branch_max_drawdowns"][branch_index], 0.0)
            assert np.isclose(info["branch_costs"][branch_index], 0.0)
            continue

        expected_raw = float(np.dot(branch_weights[1:], current_returns))
        expected_reward = float(np.log1p(expected_raw))
        expected_max_drawdown = max(0.0, -expected_raw)
        expected_violation = max(
            0.0,
            expected_max_drawdown - float(info["effective_drawdown_budget"]),
        )
        expected_cost = expected_violation**2 / env.config.drawdown_cost_scale
        assert np.isclose(info["branch_rewards"][branch_index], expected_reward)
        assert np.isclose(info["branch_transaction_costs"][branch_index], 0.0)
        assert np.isclose(info["branch_max_drawdowns"][branch_index], expected_max_drawdown)
        assert np.isclose(info["branch_costs"][branch_index], expected_cost)


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

    assert info["drawdown_benchmark_mode"] == "true_equal_weight"
    assert np.isclose(info["benchmark_raw_return"], expected_raw_return)
    assert np.isclose(info["benchmark_turnover"], expected_turnover)
    assert np.isclose(info["benchmark_transaction_cost"], expected_transaction_cost)
    assert np.isclose(info["benchmark_net_return"], expected_net_return)
    assert np.isclose(info["benchmark_current_drawdown"], expected_max_drawdown)
    assert np.isclose(info["benchmark_max_drawdown"], expected_max_drawdown)
    assert np.isclose(info["effective_drawdown_budget"], expected_budget)


def test_constrained_neutral_benchmark_uses_neutral_action_weights() -> None:
    env = build_env(
        action_mode="simplex_decomposition",
        drawdown_benchmark_mode="constrained_neutral",
        drawdown_budget_floor=0.001,
    )
    neutral_weights = env._weights_from_action(env.neutral_action())
    env.reset(options={"start_index": 4})

    _, _, _, _, info = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))

    expected_raw_return = float(
        np.dot(neutral_weights[1:], np.array([-0.02, -0.01], dtype=np.float32))
    )
    expected_turnover = float(
        np.sum(np.abs(neutral_weights - np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    )

    assert info["drawdown_benchmark_mode"] == "constrained_neutral"
    np.testing.assert_allclose(info["benchmark_weights"], neutral_weights)
    assert np.isclose(info["benchmark_raw_return"], expected_raw_return)
    assert np.isclose(info["benchmark_turnover"], expected_turnover)
    assert info["allocation_constraint_1_weight"] >= info["allocation_constraint_1_min_weight"]
    assert info["allocation_constraint_2_weight"] >= info["allocation_constraint_2_min_weight"]


def test_c3_constrained_neutral_benchmark_matches_project_simplex_weights() -> None:
    env = build_project_simplex_env()

    expected_weights = np.array(
        [0.125, 0.175, 0.175, 0.025, 0.375, 0.125],
        dtype=np.float32,
    )

    np.testing.assert_allclose(env.benchmark_weights(), expected_weights, atol=1e-7)
    assert not np.allclose(
        env.benchmark_weights(),
        np.full(6, 1.0 / 6.0, dtype=np.float32),
    )
    assert np.isclose(np.sum(env.benchmark_weights()[[1, 2, 4]]), 0.725)
    assert np.isclose(np.sum(env.benchmark_weights()[[0, 4, 5]]), 0.625)


def test_softmax_constrained_neutral_benchmark_uses_feasible_caosd_weights() -> None:
    env = build_env(
        drawdown_benchmark_mode="constrained_neutral",
        initial_portfolio_mode="constrained_neutral",
    )

    expected_weights = np.array([0.30, 0.65, 0.05], dtype=np.float32)

    np.testing.assert_allclose(env.benchmark_weights(), expected_weights, atol=1e-7)
    np.testing.assert_allclose(
        env.initial_portfolio_weights(),
        expected_weights,
        atol=1e-7,
    )
    assert not np.allclose(
        env.benchmark_weights(),
        np.full(3, 1.0 / 3.0, dtype=np.float32),
    )
    assert np.sum(env.benchmark_weights()[[0]]) >= 0.25
    assert np.sum(env.benchmark_weights()[[1]]) >= 0.60

    env.reset(options={"start_index": 2})
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    assert info["drawdown_benchmark_mode"] == "constrained_neutral"
    assert np.isclose(info["benchmark_turnover"], 0.0)


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


def test_allocation_constraint_mode_uses_scaled_allocation_violation_cost() -> None:
    env = build_env(
        constraint_mode="allocation",
        allocation_constraint_cost_scale=20.0,
    )
    env.reset(options={"start_index": 2})
    action = np.zeros(3, dtype=np.float32)

    _, _, _, _, info = env.step(action)

    expected_raw = (
        info["allocation_constraint_1_violation_cost"]
        + info["allocation_constraint_2_violation_cost"]
    )
    expected_scaled = expected_raw / 20.0
    assert info["allocation_constraint_raw_cost"] > 0.0
    assert np.isclose(info["allocation_constraint_raw_cost"], expected_raw)
    assert np.isclose(info["allocation_constraint_cost"], expected_scaled)
    assert np.isclose(info["constraint_cost"], expected_scaled)


def test_allocation_drawdown_mode_combines_scaled_allocation_and_drawdown_costs() -> None:
    env = build_env(
        constraint_mode="allocation_drawdown",
        allocation_constraint_cost_scale=20.0,
        combined_drawdown_cost_weight=0.25,
    )
    env.reset(options={"start_index": 4})
    target_weights = np.array([0.02, 0.97, 0.01], dtype=np.float32)

    _, _, _, _, info = env.step(np.log(target_weights).astype(np.float32))

    expected = (
        info["allocation_constraint_cost"]
        + 0.25 * info["drawdown_constraint_cost"]
    )
    assert info["allocation_constraint_cost"] > 0.0
    assert info["drawdown_constraint_cost"] >= 0.0
    assert info["combined_drawdown_cost_weight"] == 0.25
    assert np.isclose(info["allocation_drawdown_constraint_cost"], expected)
    assert np.isclose(info["constraint_cost"], expected)


def test_drawdown_constraint_mode_keeps_allocation_cost_diagnostic_only() -> None:
    env = build_env(
        constraint_mode="max_drawdown",
        allocation_constraint_cost_scale=20.0,
    )
    env.reset(options={"start_index": 2})

    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    assert info["allocation_constraint_cost"] > 0.0
    assert np.isclose(info["constraint_cost"], info["drawdown_constraint_cost"])


def test_constraint_preset_resolution_changes_bounds() -> None:
    env = build_env(active_constraint_preset="c3")
    resolved = env.resolved_constraint_preset()
    assert resolved["preset_name"] == "c3"
    assert np.isclose(resolved["allocation_constraint_1_min_weight"], 0.30)
    assert np.isclose(resolved["allocation_constraint_2_min_weight"], 0.50)


def test_simplex_decomposition_action_mode_uses_compact_action_shape() -> None:
    env = build_env(action_mode="simplex_decomposition")

    assert env.action_space.shape == (5,)


def test_simplex_decomposition_rejects_wrong_action_shape() -> None:
    env = build_env(action_mode="simplex_decomposition")
    env.reset(options={"start_index": 2})

    try:
        env.step(np.zeros(3, dtype=np.float32))
    except ValueError as error:
        assert "simplex-decomposition action shape" in str(error)
    else:
        raise AssertionError("Expected simplex decomposition to reject wrong action shape.")


def test_simplex_decomposition_zero_action_satisfies_allocation_constraints() -> None:
    env = build_env(action_mode="simplex_decomposition")
    env.reset(options={"start_index": 2})

    _, _, _, _, info = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))

    assert np.isclose(info["weights"].sum(), 1.0)
    assert info["allocation_constraint_1_weight"] >= info["allocation_constraint_1_min_weight"]
    assert info["allocation_constraint_2_weight"] >= info["allocation_constraint_2_min_weight"]
    assert np.isclose(info["allocation_constraint_1_violation_cost"], 0.0)
    assert np.isclose(info["allocation_constraint_2_violation_cost"], 0.0)
    assert np.isclose(info["allocation_constraint_cost"], 0.0)
    assert "simplex_z1" in info
    assert "simplex_z2" in info
    assert "simplex_z3" in info
    assert "simplex_z4" in info


def test_simplex_decomposition_branch_weights_action_format() -> None:
    env = build_env(
        action_mode="simplex_decomposition",
        simplex_action_format="branch_weights",
    )
    env.reset(options={"start_index": 2})
    action = env.neutral_action()

    _, _, _, _, info = env.step(action)

    assert action.shape == env.action_space.shape
    assert np.isclose(info["weights"].sum(), 1.0)
    assert info["allocation_constraint_1_weight"] >= info["allocation_constraint_1_min_weight"]
    assert info["allocation_constraint_2_weight"] >= info["allocation_constraint_2_min_weight"]


def test_simplex_decomposition_branch_weights_reject_negative_values() -> None:
    env = build_env(
        action_mode="simplex_decomposition",
        simplex_action_format="branch_weights",
    )
    env.reset(options={"start_index": 2})
    action = env.neutral_action()
    action[1] = -0.1

    try:
        env.step(action)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("Expected negative branch weights to be rejected.")


def test_reset_is_deterministic_when_start_index_is_fixed() -> None:
    env = build_env()
    first_obs, first_info = env.reset(options={"start_index": 2})
    second_obs, second_info = env.reset(options={"start_index": 2})
    np.testing.assert_allclose(first_obs, second_obs)
    assert first_info["start_index"] == second_info["start_index"] == 2
