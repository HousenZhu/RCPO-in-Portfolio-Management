from pathlib import Path

import numpy as np
import pytest

from rcpo_portfolio.config import EvaluationConfig, load_config
from rcpo_portfolio.market import (
    generate_continuation_splits,
    generate_train_markets,
    resolve_continuation_anchor,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "v3.3_experiment2_8assets"


V33_CASES = [
    ("rcpo_gaussian_m8_s5040_e3", 8, 5040, 3, "simplex_autoregressive_gaussian"),
    ("rcpo_gaussian_m2_s5040_e3", 2, 5040, 3, "simplex_autoregressive_gaussian"),
    ("rcpo_gaussian_m8_s10080_e3", 8, 10080, 3, "simplex_autoregressive_gaussian"),
    ("rcpo_gaussian_m2_s10080_e3", 2, 10080, 3, "simplex_autoregressive_gaussian"),
    ("rcpo_gaussian_m2_s20160_e3", 2, 20160, 3, "simplex_autoregressive_gaussian"),
    ("rcpo_gaussian_m2_s10080_e4", 2, 10080, 4, "simplex_autoregressive_gaussian"),
    ("ppo_gaussian_m2_s10080_e4", 2, 10080, 4, "simplex_autoregressive_gaussian"),
    ("rcpo_dirichlet_m2_s10080_e4", 2, 10080, 4, "simplex_autoregressive_dirichlet"),
    ("rcpo_allocation_relative_drawdown_m2_s10080_e3", 2, 10080, 3, "flat_gaussian"),
    ("cf_reward_gaussian_m2_s10080_e3", 2, 10080, 3, "simplex_autoregressive_gaussian"),
    ("cf_reward_dirichlet_m2_s10080_e4", 2, 10080, 4, "simplex_autoregressive_dirichlet"),
]


@pytest.mark.parametrize("name,count,steps,epochs,architecture", V33_CASES)
def test_v33_configs(
    name: str, count: int, steps: int, epochs: int, architecture: str
) -> None:
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    optimizer = config.ppo if name.startswith("ppo_") else config.optimization
    assert config.market.train_market_count == count
    assert config.market.train_steps == steps
    assert config.environment.episode_length == 252
    assert config.evaluation.validation_branch_count == 20
    assert config.evaluation.test_branch_count == 50
    assert config.evaluation.validation_seed_offset == 100_000
    assert config.evaluation.test_seed_offset == 200_000
    assert config.evaluation.continuation_anchor_mode == "fixed_market"
    assert config.evaluation.fixed_anchor_seed == 90_000
    assert config.evaluation.fixed_anchor_steps == 5_040
    assert optimizer.total_updates == 60_000
    assert optimizer.epochs == epochs
    assert config.network.policy_architecture == architecture


def test_v33_fixed_anchor_is_independent_of_training_market_design() -> None:
    short = load_config(CONFIG_DIR / "rcpo_gaussian_m2_s5040_e3.yaml")
    long = load_config(CONFIG_DIR / "rcpo_gaussian_m8_s10080_e3.yaml")
    seed = 0
    short_train = generate_train_markets(short.market, seed)[0]
    long_train = generate_train_markets(long.market, seed)[0]
    short_anchor = resolve_continuation_anchor(
        short.market,
        short_train,
        mode=short.evaluation.continuation_anchor_mode,
        fixed_seed=short.evaluation.fixed_anchor_seed,
        fixed_steps=short.evaluation.fixed_anchor_steps,
    )
    long_anchor = resolve_continuation_anchor(
        long.market,
        long_train,
        mode=long.evaluation.continuation_anchor_mode,
        fixed_seed=long.evaluation.fixed_anchor_seed,
        fixed_steps=long.evaluation.fixed_anchor_steps,
    )
    np.testing.assert_array_equal(short_anchor.risky_returns, long_anchor.risky_returns)
    short_validation = generate_continuation_splits(
        short.market,
        short_anchor,
        short.market.validation_steps,
        seed + short.evaluation.validation_seed_offset,
        2,
    )
    long_validation = generate_continuation_splits(
        long.market,
        long_anchor,
        long.market.validation_steps,
        seed + long.evaluation.validation_seed_offset,
        2,
    )
    for left, right in zip(short_validation, long_validation, strict=True):
        np.testing.assert_array_equal(left.risky_returns, right.risky_returns)


def test_legacy_evaluation_defaults_remain_compatible() -> None:
    defaults = EvaluationConfig()
    assert defaults.continuation_anchor_mode == "train_market"
    assert defaults.validation_seed_offset == 10_000
    assert defaults.test_seed_offset == 20_000
