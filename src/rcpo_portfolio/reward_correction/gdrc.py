from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from ..config import RewardCorrectionConfig
from ..devices import move_optimizer_state_to_device
from .base import RewardCorrectionOutput, RewardCorrector
from .drc import (
    RewardDistributionCritic,
    apply_correction_delta,
    correction_inputs,
    ordinal_cross_entropy,
    reward_labels,
)


class GDRCRewardCorrector(RewardCorrector):
    mode = "gdrc"

    def __init__(
        self,
        config: RewardCorrectionConfig,
        obs_dim: int,
        action_dim: int,
        range_window_size: int,
    ) -> None:
        self.config = config
        self.reward_min = float(config.reward_min)
        self.reward_max = float(config.reward_max)
        self.bin_counts = [int(candidate) for candidate in config.gdrc_candidate_bins]
        input_dim = obs_dim + action_dim + obs_dim
        self.models = torch.nn.ModuleList(
            [
                RewardDistributionCritic(
                    input_dim=input_dim,
                    hidden_sizes=list(config.hidden_sizes),
                    num_bins=num_bins,
                )
                for num_bins in self.bin_counts
            ]
        )
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            for model in self.models
        ]
        self.votes = np.zeros(len(self.bin_counts), dtype=np.float32)
        self.selected_index = len(self.bin_counts) - 1
        self.reward_history: deque[float] = deque(maxlen=max(1, int(range_window_size)))

    def _update_reward_range(self, observed_rewards: torch.Tensor) -> None:
        for value in observed_rewards.detach().cpu().numpy().tolist():
            self.reward_history.append(float(value))
        if not self.reward_history:
            return
        low, high = self.config.gdrc_range_percentiles
        data = np.asarray(self.reward_history, dtype=np.float32)
        reward_min = float(np.percentile(data, low))
        reward_max = float(np.percentile(data, high))
        if reward_max <= reward_min + 1e-8:
            center = 0.5 * (reward_min + reward_max)
            fallback_width = max(
                1e-3,
                0.5 * (float(self.config.reward_max) - float(self.config.reward_min)),
            )
            reward_min = center - fallback_width
            reward_max = center + fallback_width
        self.reward_min = reward_min
        self.reward_max = reward_max

    def _loss_for_model(
        self,
        model: RewardDistributionCritic,
        inputs: torch.Tensor,
        observed_rewards: torch.Tensor,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        labels, clamp_mask, bin_width = reward_labels(
            observed_rewards,
            self.reward_min,
            self.reward_max,
            num_bins,
        )
        logits = model(inputs)
        return ordinal_cross_entropy(logits, labels), labels, clamp_mask, bin_width

    def _select_candidate(self, oce_values: list[float]) -> None:
        epoch_choice = len(oce_values) - 1
        previous_increase = oce_values[0]
        for index in range(1, len(oce_values)):
            increase = oce_values[index] - oce_values[index - 1]
            if previous_increase >= increase:
                previous_increase = increase
            else:
                epoch_choice = index - 1
                break
        self.votes *= float(self.config.gdrc_vote_decay)
        self.votes[epoch_choice] += 1.0
        self.selected_index = int(np.argmax(self.votes))

    def update_and_correct(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        observed_rewards: torch.Tensor,
    ) -> RewardCorrectionOutput:
        self._update_reward_range(observed_rewards)
        inputs = correction_inputs(observations, actions, next_observations)
        train_epochs = int(self.config.train_epochs_per_update)
        oce_values: list[float] = []
        clamp_rates: list[float] = []

        for model, optimizer, num_bins in zip(self.models, self.optimizers, self.bin_counts):
            loss = torch.tensor(0.0, device=observed_rewards.device)
            labels, clamp_mask, _ = reward_labels(
                observed_rewards, self.reward_min, self.reward_max, num_bins
            )
            for _ in range(train_epochs):
                optimizer.zero_grad()
                loss, labels, clamp_mask, _ = self._loss_for_model(
                    model, inputs, observed_rewards, num_bins
                )
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                if train_epochs == 0:
                    logits = model(inputs)
                    loss = ordinal_cross_entropy(logits, labels)
                oce_values.append(float(loss.detach().item()))
                clamp_rates.append(float(clamp_mask.float().mean().item()))

        self._select_candidate(oce_values)
        selected_model = self.models[self.selected_index]
        selected_bins = self.bin_counts[self.selected_index]
        labels, clamp_mask, bin_width = reward_labels(
            observed_rewards,
            self.reward_min,
            self.reward_max,
            selected_bins,
        )
        with torch.no_grad():
            logits = selected_model(inputs)
            predicted_labels = torch.argmax(logits, dim=-1)
            raw_delta = (predicted_labels - labels).float() * float(bin_width)
            corrected_rewards, raw_delta, effective_delta = apply_correction_delta(
                observed_rewards,
                raw_delta,
                self.config,
            )
            selected_oce = ordinal_cross_entropy(logits, labels)

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
                "reward_correction_oce": float(selected_oce.detach().item()),
                "reward_correction_clamp_rate": float(clamp_mask.float().mean().item()),
                "gdrc_selected_bins": int(selected_bins),
                "gdrc_candidate_bins": list(self.bin_counts),
                "gdrc_reward_min": float(self.reward_min),
                "gdrc_reward_max": float(self.reward_max),
                "gdrc_oce_mean": float(np.mean(oce_values)) if oce_values else 0.0,
                "gdrc_vote_max": float(np.max(self.votes)) if len(self.votes) else 0.0,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "bin_counts": self.bin_counts,
            "selected_index": self.selected_index,
            "votes": self.votes.tolist(),
            "reward_history": list(self.reward_history),
            "models": [model.state_dict() for model in self.models],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.reward_min = float(state.get("reward_min", self.reward_min))
        self.reward_max = float(state.get("reward_max", self.reward_max))
        self.selected_index = int(state.get("selected_index", self.selected_index))
        if "votes" in state:
            self.votes = np.asarray(state["votes"], dtype=np.float32)
        self.reward_history.clear()
        for value in state.get("reward_history", []):
            self.reward_history.append(float(value))
        for model, model_state in zip(self.models, state.get("models", [])):
            model.load_state_dict(model_state)
        for optimizer, optimizer_state in zip(self.optimizers, state.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(
                optimizer,
                next(self.models.parameters()).device,
            )

    def to(self, device: torch.device) -> GDRCRewardCorrector:
        self.models.to(device)
        for optimizer in self.optimizers:
            move_optimizer_state_to_device(optimizer, device)
        return self
