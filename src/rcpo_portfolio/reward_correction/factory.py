from __future__ import annotations

from ..config import RewardCorrectionConfig
from .base import NoRewardCorrector, RewardCorrector
from .drc import DRCRewardCorrector
from .gdrc import GDRCRewardCorrector


def build_reward_corrector(
    config: RewardCorrectionConfig,
    obs_dim: int,
    action_dim: int,
    rollout_steps: int,
) -> RewardCorrector:
    if config.mode == "none":
        return NoRewardCorrector()
    if config.mode == "drc":
        return DRCRewardCorrector(config=config, obs_dim=obs_dim, action_dim=action_dim)
    if config.mode == "gdrc":
        range_window_size = max(1, int(config.gdrc_range_window_updates) * int(rollout_steps))
        return GDRCRewardCorrector(
            config=config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            range_window_size=range_window_size,
        )
    raise ValueError(f"Unsupported reward correction mode: {config.mode}")
