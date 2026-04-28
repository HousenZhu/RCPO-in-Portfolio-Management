from __future__ import annotations

import sys

import pytest
import torch

import train
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


def test_rcpo_constraint_flags_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo"])
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo", "--constraint-drawdown"])
    args = train.parse_args()
    assert args.constraint_drawdown


def test_constraint_flags_are_rejected_for_non_rcpo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "ppo_unconstrained", "--constraint-drawdown"],
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
