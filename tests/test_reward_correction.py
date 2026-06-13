from __future__ import annotations

import torch

import pytest

from rcpo_portfolio.config import (
    ProjectConfig,
    RewardCorrectionConfig,
    load_config,
    validate_reward_correction_settings,
)
from rcpo_portfolio.reward_correction.drc import (
    DRCRewardCorrector,
    RewardDistributionCritic,
    ordinal_cross_entropy,
    reward_labels,
)
from rcpo_portfolio.reward_correction.gdrc import GDRCRewardCorrector


def test_reward_labels_clamp_out_of_range_rewards() -> None:
    rewards = torch.tensor([-0.2, -0.05, 0.0, 0.049, 0.2])
    labels, clamp_mask, bin_width = reward_labels(rewards, -0.05, 0.05, 5)

    assert torch.equal(labels, torch.tensor([0, 0, 2, 4, 4]))
    assert torch.equal(clamp_mask, torch.tensor([True, False, False, False, True]))
    assert bin_width == 0.02


def test_ordinal_cross_entropy_penalizes_farther_predictions() -> None:
    labels = torch.tensor([2])
    close_logits = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0]])
    far_logits = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])

    assert ordinal_cross_entropy(far_logits, labels) > ordinal_cross_entropy(close_logits, labels)


def test_drc_forward_returns_probabilities() -> None:
    model = RewardDistributionCritic(input_dim=4, hidden_sizes=[8], num_bins=5)
    probabilities = model.probabilities(torch.zeros((3, 4)))

    assert probabilities.shape == (3, 5)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))


def test_reward_correction_config_loads_fine_bin_defaults() -> None:
    config = load_config("configs/default.yaml")

    assert config.reward_correction.num_bins == 48
    assert config.reward_correction.gdrc_candidate_bins == [48, 64]
    assert config.reward_correction.correction_coef == pytest.approx(0.50)
    assert config.reward_correction.correction_delta_clip == pytest.approx(0.0015)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("correction_coef", -0.1, "correction_coef"),
        ("correction_delta_clip", -0.1, "correction_delta_clip"),
        ("gdrc_candidate_bins", [], "gdrc_candidate_bins"),
        ("gdrc_candidate_bins", [1], "gdrc_candidate_bins"),
    ],
)
def test_reward_correction_config_rejects_invalid_stabilization_settings(
    field_name: str,
    value,
    message: str,
) -> None:
    config = ProjectConfig()
    setattr(config.reward_correction, field_name, value)

    with pytest.raises(ValueError, match=message):
        validate_reward_correction_settings(config)


def test_standalone_branch_credit_rejects_reward_correction() -> None:
    config = ProjectConfig()
    config.network.branch_credit_mode = "standalone"
    config.reward_correction.mode = "gdrc"

    with pytest.raises(ValueError, match="requires reward_correction.mode='none'"):
        validate_reward_correction_settings(config)


def test_drc_corrected_rewards_are_scaled_and_clipped() -> None:
    config = RewardCorrectionConfig(
        mode="drc",
        reward_min=-0.05,
        reward_max=0.05,
        hidden_sizes=[],
        learning_rate=1e-3,
        train_epochs_per_update=0,
        num_bins=5,
        correction_coef=0.5,
        correction_delta_clip=0.0015,
    )
    corrector = DRCRewardCorrector(config=config, obs_dim=1, action_dim=1)
    with torch.no_grad():
        linear = corrector.model.net[0]
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0]))

    observed_rewards = torch.tensor([-0.01, 0.01])
    output = corrector.update_and_correct(
        observations=torch.zeros((2, 1)),
        actions=torch.zeros((2, 1)),
        next_observations=torch.zeros((2, 1)),
        observed_rewards=observed_rewards,
    )

    labels, _, bin_width = reward_labels(observed_rewards, -0.05, 0.05, 5)
    raw_delta = (torch.tensor([3, 3]) - labels).float() * bin_width
    effective_delta = torch.clamp(0.5 * raw_delta, min=-0.0015, max=0.0015)
    expected = observed_rewards + effective_delta
    torch.testing.assert_close(output.corrected_rewards, expected)
    assert output.metrics["reward_correction_raw_delta_abs_mean"] > output.metrics[
        "reward_correction_effective_delta_abs_mean"
    ]
    assert output.metrics["reward_correction_delta_abs_mean"] == output.metrics[
        "reward_correction_effective_delta_abs_mean"
    ]


def test_gdrc_updates_range_and_selects_valid_candidate() -> None:
    config = RewardCorrectionConfig(
        mode="gdrc",
        reward_min=-0.05,
        reward_max=0.05,
        hidden_sizes=[8],
        learning_rate=1e-3,
        train_epochs_per_update=1,
        gdrc_candidate_bins=[48, 64],
        gdrc_range_percentiles=[1, 99],
    )
    corrector = GDRCRewardCorrector(
        config=config,
        obs_dim=2,
        action_dim=1,
        range_window_size=16,
    )
    output = corrector.update_and_correct(
        observations=torch.zeros((8, 2)),
        actions=torch.zeros((8, 1)),
        next_observations=torch.ones((8, 2)),
        observed_rewards=torch.linspace(-0.02, 0.03, 8),
    )

    assert corrector.bin_counts == [48, 64]
    assert output.metrics["gdrc_selected_bins"] in {48, 64}
    assert output.metrics["gdrc_candidate_bins"] == [48, 64]
    assert output.metrics["gdrc_reward_min"] < output.metrics["gdrc_reward_max"]
    assert output.metrics["reward_correction_effective_delta_abs_mean"] <= 0.0015
    state = corrector.state_dict()
    assert state["selected_index"] >= 0
    assert state["bin_counts"] == [48, 64]
    assert len(state["votes"]) == 2
