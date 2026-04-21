from __future__ import annotations

import pytest
import torch

from rcpo_portfolio.config import ProjectConfig, sync_rcpo_constraint_settings
from rcpo_portfolio.devices import resolve_device
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.market import generate_market_split
from rcpo_portfolio.models import ActorCritic
from rcpo_portfolio.reward_correction import NoRewardCorrector
from rcpo_portfolio.rollouts import collect_rollout


def test_resolve_device_for_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_uses_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_cuda_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_collect_rollout_places_batch_on_selected_device() -> None:
    config = ProjectConfig()
    config.market.lookback = 2
    config.market.train_steps = 32
    config.environment.episode_length = 8
    config.optimization.rollout_steps = 8
    sync_rcpo_constraint_settings(config)
    market = generate_market_split(config.market, steps=config.market.train_steps, seed=123)
    env = PortfolioEnv(config.environment, market, config.market, seed=123)
    model = ActorCritic(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config.network,
    )
    device = torch.device("cpu")

    batch = collect_rollout(
        env=env,
        model=model,
        optimization=config.optimization,
        reward_corrector=NoRewardCorrector(),
        device=device,
    )

    assert batch.observations.device == device
    assert batch.actions.device == device
    assert batch.reward_advantages.device == device

