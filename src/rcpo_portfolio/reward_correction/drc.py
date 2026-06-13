from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..config import RewardCorrectionConfig
from ..devices import move_optimizer_state_to_device
from .base import RewardCorrectionOutput, RewardCorrector


def _activation() -> nn.Module:
    return nn.Tanh()


class RewardDistributionCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], num_bins: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(_activation())
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, num_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(inputs), dim=-1)


def reward_labels(
    rewards: torch.Tensor,
    reward_min: float,
    reward_max: float,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if reward_max <= reward_min:
        raise ValueError("reward_max must be greater than reward_min.")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    bin_width = (float(reward_max) - float(reward_min)) / float(num_bins)
    raw_labels = torch.floor((rewards - float(reward_min)) / bin_width).long()
    clamped_labels = torch.clamp(raw_labels, min=0, max=num_bins - 1)
    clamp_mask = raw_labels != clamped_labels
    return clamped_labels, clamp_mask, bin_width


def ordinal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    if num_bins < 2:
        raise ValueError("ordinal_cross_entropy requires at least two bins.")
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        predicted_labels = torch.argmax(logits, dim=-1)
        distance = torch.abs(predicted_labels - labels).float()
        weights = 1.0 + distance / float(num_bins - 1)
    nll = F.nll_loss(log_probs, labels, reduction="none")
    return torch.mean(weights * nll)


def correction_inputs(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([observations, actions, next_observations], dim=-1)


def apply_correction_delta(
    observed_rewards: torch.Tensor,
    raw_delta: torch.Tensor,
    config: RewardCorrectionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scaled_delta = raw_delta * float(config.correction_coef)
    delta_clip = float(config.correction_delta_clip)
    clip_tensor = torch.as_tensor(
        delta_clip,
        dtype=scaled_delta.dtype,
        device=scaled_delta.device,
    )
    # Keep the represented floating-point value inside the configured decimal bound.
    safe_clip = torch.nextafter(clip_tensor, torch.zeros_like(clip_tensor))
    effective_delta = torch.clamp(scaled_delta, min=-safe_clip, max=safe_clip)
    corrected_rewards = observed_rewards + effective_delta
    return corrected_rewards, raw_delta, effective_delta


class DRCRewardCorrector(RewardCorrector):
    mode = "drc"

    def __init__(self, config: RewardCorrectionConfig, obs_dim: int, action_dim: int) -> None:
        self.config = config
        self.num_bins = int(config.num_bins)
        self.reward_min = float(config.reward_min)
        self.reward_max = float(config.reward_max)
        input_dim = obs_dim + action_dim + obs_dim
        self.model = RewardDistributionCritic(
            input_dim=input_dim,
            hidden_sizes=list(config.hidden_sizes),
            num_bins=self.num_bins,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _loss_and_labels(
        self,
        inputs: torch.Tensor,
        observed_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        labels, clamp_mask, bin_width = reward_labels(
            observed_rewards,
            self.reward_min,
            self.reward_max,
            self.num_bins,
        )
        logits = self.model(inputs)
        return ordinal_cross_entropy(logits, labels), labels, clamp_mask, bin_width

    def update_and_correct(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        observed_rewards: torch.Tensor,
    ) -> RewardCorrectionOutput:
        inputs = correction_inputs(observations, actions, next_observations)
        train_epochs = int(self.config.train_epochs_per_update)
        loss = torch.tensor(0.0, device=observed_rewards.device)
        labels, clamp_mask, bin_width = reward_labels(
            observed_rewards,
            self.reward_min,
            self.reward_max,
            self.num_bins,
        )
        for _ in range(train_epochs):
            self.optimizer.zero_grad()
            loss, labels, clamp_mask, bin_width = self._loss_and_labels(inputs, observed_rewards)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            logits = self.model(inputs)
            if train_epochs == 0:
                loss = ordinal_cross_entropy(logits, labels)
            predicted_labels = torch.argmax(logits, dim=-1)
            raw_delta = (predicted_labels - labels).float() * float(bin_width)
            corrected_rewards, raw_delta, effective_delta = apply_correction_delta(
                observed_rewards,
                raw_delta,
                self.config,
            )

        return RewardCorrectionOutput(
            corrected_rewards=corrected_rewards.detach(),
            metrics={
                "reward_correction_mode": self.mode,
                "observed_reward_mean": float(observed_rewards.mean().item()),
                "corrected_reward_mean": float(corrected_rewards.mean().item()),
                "reward_correction_delta_mean": float(effective_delta.mean().item()),
                "reward_correction_delta_abs_mean": float(
                    min(
                        torch.abs(effective_delta).mean().item(),
                        float(self.config.correction_delta_clip),
                    )
                ),
                "reward_correction_raw_delta_abs_mean": float(
                    torch.abs(raw_delta).mean().item()
                ),
                "reward_correction_effective_delta_abs_mean": float(
                    min(
                        torch.abs(effective_delta).mean().item(),
                        float(self.config.correction_delta_clip),
                    )
                ),
                "reward_correction_coef": float(self.config.correction_coef),
                "reward_correction_delta_clip": float(self.config.correction_delta_clip),
                "reward_correction_oce": float(loss.detach().item()),
                "reward_correction_clamp_rate": float(clamp_mask.float().mean().item()),
                "gdrc_selected_bins": 0,
                "gdrc_candidate_bins": [],
                "gdrc_reward_min": self.reward_min,
                "gdrc_reward_max": self.reward_max,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "num_bins": self.num_bins,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            move_optimizer_state_to_device(
                self.optimizer,
                next(self.model.parameters()).device,
            )

    def to(self, device: torch.device) -> DRCRewardCorrector:
        self.model.to(device)
        move_optimizer_state_to_device(self.optimizer, device)
        return self
