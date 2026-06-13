from __future__ import annotations

import numpy as np
import pytest
import torch

from rcpo_portfolio.config import ProjectConfig, load_config, validate_reward_noise_settings
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.market import generate_market_split
from rcpo_portfolio.models import ActorCritic
from rcpo_portfolio.reward_correction import NoRewardCorrector
from rcpo_portfolio.rollouts import collect_rollout


def _small_config() -> ProjectConfig:
    config = ProjectConfig()
    config.market.lookback = 2
    config.market.train_steps = 64
    config.environment.episode_length = 8
    config.optimization.rollout_steps = 16
    config.optimization.epochs = 1
    config.optimization.minibatch_size = 8
    config.reward_noise.std = 0.003
    return config


def _rollout(config: ProjectConfig, torch_seed: int, noise_seed: int | None = None):
    market = generate_market_split(config.market, steps=config.market.train_steps, seed=123)
    env = PortfolioEnv(config.environment, market, config.market, seed=123)
    torch.manual_seed(torch_seed)
    model = ActorCritic(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config.network,
    )
    torch.manual_seed(torch_seed + 1)
    return collect_rollout(
        env=env,
        model=model,
        optimization=config.optimization,
        reward_corrector=NoRewardCorrector(),
        device=torch.device("cpu"),
        reward_noise_config=config.reward_noise,
        reward_noise_rng=(
            np.random.default_rng(noise_seed) if noise_seed is not None else None
        ),
    )


def test_noisy_reward_config_loads_and_validates() -> None:
    default_config = load_config("configs/default.yaml")

    assert default_config.reward_noise.mode == "gaussian"
    assert default_config.reward_noise.std == pytest.approx(0.003)

    default_config.network.branch_credit_mode = "global"
    default_config.reward_noise.enabled = True
    validate_reward_noise_settings(default_config)

    default_config.reward_noise.std = -0.001
    with pytest.raises(ValueError, match="reward_noise.std"):
        validate_reward_noise_settings(default_config)


def test_standalone_branch_credit_rejects_reward_noise() -> None:
    config = ProjectConfig()
    config.network.branch_credit_mode = "standalone"
    config.reward_noise.enabled = True

    with pytest.raises(ValueError, match="requires clean rewards"):
        validate_reward_noise_settings(config)


def test_clean_reward_noise_mode_keeps_observed_rewards_clean() -> None:
    config = _small_config()
    config.reward_noise.enabled = False

    batch = _rollout(config, torch_seed=7)

    torch.testing.assert_close(batch.observed_rewards, batch.true_rewards)
    torch.testing.assert_close(batch.rewards, batch.true_rewards)
    assert batch.info_summary["reward_noise_enabled"] == 0
    assert batch.info_summary["batch_reward_noise_std"] == 0.0


def test_gaussian_reward_noise_changes_only_observed_reward_stream() -> None:
    clean_config = _small_config()
    clean_config.reward_noise.enabled = False
    noisy_config = _small_config()
    noisy_config.reward_noise.enabled = True

    clean = _rollout(clean_config, torch_seed=11)
    noisy = _rollout(noisy_config, torch_seed=11, noise_seed=19)

    torch.testing.assert_close(clean.true_rewards, noisy.true_rewards)
    torch.testing.assert_close(clean.costs, noisy.costs)
    assert not torch.allclose(noisy.observed_rewards, noisy.true_rewards)
    torch.testing.assert_close(noisy.rewards, noisy.observed_rewards)
    assert noisy.info_summary["reward_noise_enabled"] == 1
    assert noisy.info_summary["reward_noise_std"] == pytest.approx(0.003)
    assert noisy.info_summary["batch_reward_noise_std"] > 0.0


def test_gaussian_reward_noise_is_deterministic_for_same_seed() -> None:
    config = _small_config()
    config.reward_noise.enabled = True

    first = _rollout(config, torch_seed=13, noise_seed=29)
    second = _rollout(config, torch_seed=13, noise_seed=29)

    torch.testing.assert_close(first.true_rewards, second.true_rewards)
    torch.testing.assert_close(first.observed_rewards, second.observed_rewards)
