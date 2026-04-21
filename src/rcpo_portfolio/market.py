from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MarketConfig


@dataclass
class MarketSlice:
    risky_returns: np.ndarray
    regimes: np.ndarray


def _covariance_matrix(num_assets: int, vol_scale: float, correlation_value: float) -> np.ndarray:
    correlation = np.full((num_assets, num_assets), correlation_value, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    asset_multipliers = np.linspace(0.9, 1.1, num_assets)
    vol_vector = vol_scale * asset_multipliers
    return np.outer(vol_vector, vol_vector) * correlation


def _sized_vector(values: list[float], size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) >= size:
        return vector[:size]
    padded = np.zeros(size, dtype=np.float64)
    padded[: len(vector)] = vector
    return padded


def _regime_model(config: MarketConfig) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    drift_offsets = np.linspace(-0.0002, 0.0002, config.num_risky_assets)
    low_adjustment = np.zeros(config.num_risky_assets, dtype=np.float64)
    high_adjustment = np.zeros(config.num_risky_assets, dtype=np.float64)
    low_correlation = config.base_correlation
    high_correlation = config.base_correlation
    if config.enable_learnable_structure:
        low_adjustment = _sized_vector(
            config.regime_drift_adjustments.get("low_vol", []),
            config.num_risky_assets,
        )
        high_adjustment = _sized_vector(
            config.regime_drift_adjustments.get("high_vol", []),
            config.num_risky_assets,
        )
        low_correlation = config.low_vol_correlation
        high_correlation = config.high_vol_correlation
    regime_drifts = {
        0: config.low_vol_drift + drift_offsets + low_adjustment,
        1: config.high_vol_drift + drift_offsets + high_adjustment,
    }
    regime_covariances = {
        0: _covariance_matrix(
            config.num_risky_assets, config.low_vol_scale, low_correlation
        ),
        1: _covariance_matrix(
            config.num_risky_assets, config.high_vol_scale, high_correlation
        ),
    }
    return regime_drifts, regime_covariances


def _simulate_initial_regimes(config: MarketConfig, total_steps: int, rng: np.random.Generator) -> np.ndarray:
    transition = np.asarray(config.transition_matrix, dtype=np.float64)
    regimes = np.zeros(total_steps, dtype=np.int64)
    for index in range(1, total_steps):
        previous_regime = regimes[index - 1]
        regimes[index] = rng.choice(2, p=transition[previous_regime])
    return regimes


def _simulate_future_regimes(
    config: MarketConfig,
    steps: int,
    rng: np.random.Generator,
    previous_regime: int,
) -> np.ndarray:
    transition = np.asarray(config.transition_matrix, dtype=np.float64)
    regimes = np.zeros(steps, dtype=np.int64)
    current_regime = int(previous_regime)
    for index in range(steps):
        current_regime = int(rng.choice(2, p=transition[current_regime]))
        regimes[index] = current_regime
    return regimes


def _sample_returns_for_regimes(
    config: MarketConfig,
    regimes: np.ndarray,
    rng: np.random.Generator,
    initial_trend: np.ndarray | None = None,
) -> np.ndarray:
    regime_drifts, regime_covariances = _regime_model(config)

    risky_returns = np.zeros((len(regimes), config.num_risky_assets), dtype=np.float32)
    trend = (
        np.zeros(config.num_risky_assets, dtype=np.float64)
        if initial_trend is None
        else np.asarray(initial_trend, dtype=np.float64).copy()
    )
    for index, regime in enumerate(regimes):
        mean = regime_drifts[int(regime)]
        if config.enable_learnable_structure and config.momentum_strength != 0.0:
            mean = mean + config.momentum_strength * np.clip(
                trend,
                -float(config.momentum_clip),
                float(config.momentum_clip),
            )
        risky_returns[index] = rng.multivariate_normal(
            mean=mean,
            cov=regime_covariances[int(regime)],
        ).astype(np.float32)
        if config.enable_learnable_structure:
            trend = (
                float(config.momentum_decay) * trend
                + (1.0 - float(config.momentum_decay)) * risky_returns[index].astype(np.float64)
            )
    return risky_returns


def _trend_from_history(config: MarketConfig, risky_returns: np.ndarray) -> np.ndarray:
    trend = np.zeros(config.num_risky_assets, dtype=np.float64)
    if not config.enable_learnable_structure:
        return trend
    for row in risky_returns:
        trend = (
            float(config.momentum_decay) * trend
            + (1.0 - float(config.momentum_decay)) * row.astype(np.float64)
        )
    return trend


def generate_market_split(config: MarketConfig, steps: int, seed: int) -> MarketSlice:
    rng = np.random.default_rng(seed)
    total_steps = steps + config.lookback
    regimes = _simulate_initial_regimes(config, total_steps, rng)
    risky_returns = _sample_returns_for_regimes(config, regimes, rng)
    return MarketSlice(risky_returns=risky_returns, regimes=regimes)


def generate_continuation_split(
    config: MarketConfig,
    train_market: MarketSlice,
    steps: int,
    seed: int,
) -> MarketSlice:
    rng = np.random.default_rng(seed)
    if len(train_market.risky_returns) < config.lookback:
        raise ValueError("Training market is too short to provide continuation lookback.")
    lookback_returns = train_market.risky_returns[-config.lookback :]
    lookback_regimes = train_market.regimes[-config.lookback :]
    future_regimes = _simulate_future_regimes(
        config,
        steps,
        rng,
        previous_regime=int(train_market.regimes[-1]),
    )
    initial_trend = _trend_from_history(config, train_market.risky_returns)
    future_returns = _sample_returns_for_regimes(
        config,
        future_regimes,
        rng,
        initial_trend=initial_trend,
    )
    return MarketSlice(
        risky_returns=np.concatenate([lookback_returns, future_returns], axis=0),
        regimes=np.concatenate([lookback_regimes, future_regimes], axis=0),
    )


def generate_continuation_splits(
    config: MarketConfig,
    train_market: MarketSlice,
    steps: int,
    seed: int,
    count: int,
) -> list[MarketSlice]:
    return [
        generate_continuation_split(config, train_market, steps, seed + 101 * index)
        for index in range(count)
    ]


def generate_train_markets(config: MarketConfig, seed: int) -> list[MarketSlice]:
    count = max(1, int(config.train_market_count))
    return [
        generate_market_split(config, config.train_steps, seed + 11 + 9973 * index)
        for index in range(count)
    ]


def generate_market_splits(config: MarketConfig, seed: int) -> dict[str, MarketSlice]:
    train_market = generate_train_markets(config, seed)[0]
    return {
        "train": train_market,
        "validation": generate_continuation_split(
            config, train_market, config.validation_steps, seed + 23
        ),
        "test": generate_continuation_split(
            config, train_market, config.test_steps, seed + 37
        ),
    }
