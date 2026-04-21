from __future__ import annotations

import numpy as np

from rcpo_portfolio.config import MarketConfig
from rcpo_portfolio.market import (
    _regime_model,
    _sample_returns_for_regimes,
    generate_market_splits,
    generate_train_markets,
)


def test_generate_train_markets_are_deterministic_and_distinct() -> None:
    config = MarketConfig(
        num_risky_assets=3,
        lookback=2,
        train_steps=12,
        train_market_count=3,
    )

    first = generate_train_markets(config, seed=4)
    second = generate_train_markets(config, seed=4)

    assert len(first) == 3
    assert len(second) == 3
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.risky_returns, right.risky_returns)
        np.testing.assert_array_equal(left.regimes, right.regimes)
    assert not np.array_equal(first[0].risky_returns, first[1].risky_returns)


def test_validation_and_test_continue_from_anchor_train_market() -> None:
    config = MarketConfig(
        num_risky_assets=3,
        lookback=3,
        train_steps=20,
        validation_steps=8,
        test_steps=8,
        train_market_count=4,
    )

    train_markets = generate_train_markets(config, seed=9)
    splits = generate_market_splits(config, seed=9)

    np.testing.assert_allclose(splits["train"].risky_returns, train_markets[0].risky_returns)
    np.testing.assert_allclose(
        splits["validation"].risky_returns[: config.lookback],
        train_markets[0].risky_returns[-config.lookback :],
    )
    np.testing.assert_allclose(
        splits["test"].risky_returns[: config.lookback],
        train_markets[0].risky_returns[-config.lookback :],
    )


def test_learnable_structure_sets_regime_winners_and_correlations() -> None:
    config = MarketConfig()
    drifts, covariances = _regime_model(config)

    assert drifts[0][0] > drifts[0][2]
    assert drifts[0][1] > drifts[0][3]
    assert drifts[1][4] > drifts[1][0]
    low_corr = covariances[0][0, 1] / np.sqrt(covariances[0][0, 0] * covariances[0][1, 1])
    high_corr = covariances[1][0, 1] / np.sqrt(covariances[1][0, 0] * covariances[1][1, 1])
    assert high_corr > low_corr


def test_momentum_increases_drift_in_direction_of_recent_trend() -> None:
    config = MarketConfig(
        num_risky_assets=2,
        low_vol_drift=0.0,
        high_vol_drift=0.0,
        low_vol_scale=0.0,
        high_vol_scale=0.0,
        enable_learnable_structure=True,
        regime_drift_adjustments={"low_vol": [0.0, 0.0], "high_vol": [0.0, 0.0]},
        momentum_strength=0.5,
        momentum_decay=0.9,
        momentum_clip=0.01,
    )

    no_trend_returns = _sample_returns_for_regimes(
        config,
        np.array([0], dtype=np.int64),
        np.random.default_rng(1),
        initial_trend=np.array([0.0, 0.0], dtype=np.float64),
    )
    trend_returns = _sample_returns_for_regimes(
        config,
        np.array([0], dtype=np.int64),
        np.random.default_rng(1),
        initial_trend=np.array([0.02, -0.02], dtype=np.float64),
    )

    assert trend_returns[0, 0] > no_trend_returns[0, 0]
    assert trend_returns[0, 1] < no_trend_returns[0, 1]
    assert np.isclose(trend_returns[0, 0] - no_trend_returns[0, 0], 0.005)
    assert np.isclose(trend_returns[0, 1] - no_trend_returns[0, 1], -0.005)


def test_disabling_learnable_structure_uses_base_regime_model() -> None:
    config = MarketConfig(
        num_risky_assets=2,
        enable_learnable_structure=False,
        low_vol_drift=0.001,
        high_vol_drift=-0.001,
        base_correlation=0.33,
        low_vol_correlation=0.01,
        high_vol_correlation=0.99,
        regime_drift_adjustments={
            "low_vol": [1.0, 1.0],
            "high_vol": [1.0, 1.0],
        },
    )

    drifts, covariances = _regime_model(config)

    np.testing.assert_allclose(drifts[0], np.array([0.0008, 0.0012]))
    np.testing.assert_allclose(drifts[1], np.array([-0.0012, -0.0008]))
    corr = covariances[0][0, 1] / np.sqrt(covariances[0][0, 0] * covariances[0][1, 1])
    assert np.isclose(corr, 0.33)
