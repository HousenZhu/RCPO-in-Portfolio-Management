from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RewardCorrectionOutput:
    corrected_rewards: torch.Tensor
    metrics: dict[str, float | int | str]


class RewardCorrector:
    mode = "none"

    def update_and_correct(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        observed_rewards: torch.Tensor,
    ) -> RewardCorrectionOutput:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state

    def to(self, device: torch.device) -> RewardCorrector:
        del device
        return self


class NoRewardCorrector(RewardCorrector):
    mode = "none"

    def update_and_correct(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        observed_rewards: torch.Tensor,
    ) -> RewardCorrectionOutput:
        del observations, actions, next_observations
        corrected = observed_rewards.detach().clone()
        return RewardCorrectionOutput(
            corrected_rewards=corrected,
            metrics={
                "reward_correction_mode": self.mode,
                "observed_reward_mean": float(observed_rewards.mean().item()),
                "corrected_reward_mean": float(corrected.mean().item()),
                "reward_correction_delta_mean": 0.0,
                "reward_correction_delta_abs_mean": 0.0,
                "reward_correction_oce": 0.0,
                "reward_correction_clamp_rate": 0.0,
                "gdrc_selected_bins": 0,
                "gdrc_reward_min": 0.0,
                "gdrc_reward_max": 0.0,
            },
        )
