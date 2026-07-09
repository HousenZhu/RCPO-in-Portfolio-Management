from __future__ import annotations

import sys

import pytest
import torch

import train
from rcpo_portfolio.algorithms.ppo import standalone_branch_policy_loss
from rcpo_portfolio.config import ProjectConfig, load_config, sync_rcpo_constraint_settings
from rcpo_portfolio.trainer import combine_advantages, update_lagrange_multiplier


def test_lagrange_multiplier_never_becomes_negative() -> None:
    updated = update_lagrange_multiplier(lambda_value=0.1, observed_cost=0.0, alpha=1.0, learning_rate=1.0)
    assert updated == 0.0


def test_lagrange_multiplier_moves_with_constraint_gap() -> None:
    increased = update_lagrange_multiplier(lambda_value=0.2, observed_cost=0.5, alpha=0.1, learning_rate=0.5)
    stalled = update_lagrange_multiplier(lambda_value=0.2, observed_cost=0.05, alpha=0.1, learning_rate=0.5)
    assert increased > 0.2
    assert stalled < 0.2


def test_lagrange_multiplier_uses_asymmetric_learning_rates() -> None:
    increased = update_lagrange_multiplier(
        lambda_value=0.2,
        observed_cost=0.5,
        alpha=0.1,
        learning_rate_up=0.015,
        learning_rate_down=0.03,
    )
    decreased = update_lagrange_multiplier(
        lambda_value=0.2,
        observed_cost=0.05,
        alpha=0.1,
        learning_rate_up=0.015,
        learning_rate_down=0.03,
    )

    assert increased == pytest.approx(0.2 + 0.015 * 0.4)
    assert decreased == pytest.approx(0.2 + 0.03 * -0.05)


def test_combined_advantage_matches_reward_advantage_when_lambda_zero() -> None:
    reward_advantages = torch.tensor([1.0, 2.0, 3.0])
    cost_advantages = torch.tensor([0.5, 0.25, 0.75])
    combined = combine_advantages(reward_advantages, cost_advantages, lambda_value=0.0)
    assert torch.allclose(combined, reward_advantages)


def test_combined_advantage_shapes_stay_consistent() -> None:
    reward_advantages = torch.randn(8)
    cost_advantages = torch.randn(8)
    combined = combine_advantages(reward_advantages, cost_advantages, lambda_value=0.3)
    assert combined.shape == reward_advantages.shape == cost_advantages.shape


def test_standalone_branch_policy_loss_is_weighted_by_actual_caosd_mass() -> None:
    ratios = torch.tensor([[1.10, 0.90, 1.20, 0.80]], requires_grad=True)
    clipped = torch.tensor([[1.10, 0.90, 1.15, 0.85]])
    advantages = torch.tensor([[1.0, -2.0, 0.5, -1.0]])
    z_values = torch.tensor([[0.10, 0.20, 0.0, 0.70]])

    loss = standalone_branch_policy_loss(ratios, clipped, advantages, z_values)
    expected_surrogate = torch.minimum(ratios.detach() * advantages, clipped * advantages)
    expected = -torch.sum(z_values * expected_surrogate)
    assert loss.item() == pytest.approx(expected.item())

    loss.backward()
    assert ratios.grad is not None
    assert ratios.grad[0, 0].abs() > 0.0
    assert ratios.grad[0, 2] == 0.0


def test_rcpo_constraint_flags_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo"])
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo", "--constraint-drawdown"])
    args = train.parse_args()
    assert args.constraint_drawdown
    assert not args.constraint_allocation

    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo", "--constraint-allocation"])
    args = train.parse_args()
    assert args.constraint_allocation
    assert not args.constraint_drawdown

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "rcpo", "--constraint-drawdown", "--constraint-allocation"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_constraint_flags_are_rejected_for_non_rcpo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "ppo_unconstrained", "--constraint-drawdown"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "ppo_unconstrained", "--constraint-allocation"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_legacy_constraint_flags_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "rcpo", "--constraint-downside"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "rcpo", "--constraint-sortino"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_allocation_constraint_mode_requires_fixed_alpha() -> None:
    config = ProjectConfig()
    config.rcpo.constraint_mode = "allocation"
    config.rcpo.alpha = None
    with pytest.raises(ValueError, match="rcpo.alpha"):
        sync_rcpo_constraint_settings(config)


def test_allocation_penalty_config_loads() -> None:
    config = load_config("configs/rcpo_allocation_penalty.yaml")
    sync_rcpo_constraint_settings(config)

    assert config.environment.action_mode == "softmax"
    assert config.network.policy_architecture == "flat_gaussian"
    assert config.network.branch_credit_mode == "global"
    assert config.rcpo.constraint_mode == "allocation"
    assert config.environment.constraint_mode == "allocation"
    assert config.environment.drawdown_benchmark_mode == "constrained_neutral"
    assert config.rcpo.alpha == pytest.approx(0.0005)
    assert config.environment.allocation_constraint_cost_scale == pytest.approx(20.0)


def test_reward_correction_flags_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "ppo_unconstrained", "--use-drc", "--use-gdrc"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_reward_correction_flags_are_rejected_for_equal_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "equal_weight", "--use-drc"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()


def test_reward_correction_flags_are_allowed_for_ppo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "ppo_unconstrained", "--use-gdrc"],
    )
    args = train.parse_args()
    assert args.use_gdrc
