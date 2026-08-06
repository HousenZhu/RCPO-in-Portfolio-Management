from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import train
from rcpo_portfolio.algorithms.ppo import (
    standalone_branch_policy_loss,
    update_ppo_actor_critic,
)
from rcpo_portfolio.config import (
    NetworkConfig,
    ProjectConfig,
    load_config,
    sync_rcpo_constraint_settings,
)
from rcpo_portfolio.models import ActorCritic
from rcpo_portfolio.rollouts import RolloutBatch
from rcpo_portfolio.trainer import RCPOTrainer, combine_advantages, update_lagrange_multiplier


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



def test_resume_learning_rate_schedule_starts_from_checkpoint_value() -> None:
    trainer = object.__new__(RCPOTrainer)
    trainer.optimization = SimpleNamespace(
        learning_rate=1e-4,
        learning_rate_final=2e-5,
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer.optimizer = torch.optim.Adam([parameter], lr=6e-5)

    checkpoint_learning_rate = 6e-5
    first = trainer._set_learning_rate(
        update_index=0,
        total_updates=10,
        start_learning_rate=checkpoint_learning_rate,
    )
    final = trainer._set_learning_rate(
        update_index=9,
        total_updates=10,
        start_learning_rate=checkpoint_learning_rate,
    )

    assert first == pytest.approx(checkpoint_learning_rate)
    assert final == pytest.approx(2e-5)

def test_resume_metrics_reader_repairs_escaped_newline_prefix(tmp_path) -> None:
    trainer = object.__new__(RCPOTrainer)
    trainer.metrics_path = tmp_path / "metrics.jsonl"
    trainer.resume_checkpoint = tmp_path / "checkpoint.pt"
    trainer.resume_completed_updates = 1
    trainer.metrics_path.write_text(
        '{"update": 0}\n' + r"\n" + '{"update": 1}\n',
        encoding="utf-8",
    )

    rows = trainer._read_metric_rows()

    assert [row["update"] for row in rows] == [0]
    assert trainer.metrics_path.read_text(encoding="utf-8") == '{"update": 0}\n'


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


def _standalone_policy_test_batch(
    model: ActorCritic,
    observations: torch.Tensor,
) -> RolloutBatch:
    with torch.no_grad():
        output = model.get_policy_output(observations)
    count = observations.shape[0]
    branch_count = len(model.branch_sizes)
    base_advantage = torch.linspace(-1.0, 1.0, count)
    branch_advantages = torch.stack(
        [torch.roll(base_advantage, shifts=index) for index in range(branch_count)],
        dim=-1,
    )
    zeros = torch.zeros(count)
    branch_zeros = torch.zeros((count, branch_count))
    return RolloutBatch(
        observations=observations,
        actions=output.action.detach(),
        next_observations=observations.clone(),
        log_probs=output.log_prob.detach(),
        true_rewards=zeros.clone(),
        observed_rewards=zeros.clone(),
        rewards=zeros.clone(),
        costs=zeros.clone(),
        dones=zeros.clone(),
        reward_values=output.reward_value.detach(),
        cost_values=output.cost_value.detach(),
        reward_returns=output.reward_value.detach(),
        cost_returns=output.cost_value.detach(),
        reward_advantages=base_advantage,
        cost_advantages=zeros.clone(),
        branch_log_probs=output.branch_log_probs.detach(),
        branch_entropies=output.branch_entropies.detach(),
        branch_rewards=branch_zeros.clone(),
        branch_costs=branch_zeros.clone(),
        branch_z_values=torch.tensor(
            [[0.0, 0.40, 0.35, 0.25]], dtype=torch.float32
        ).expand(count, -1),
        branch_reward_values=output.branch_reward_values.detach(),
        branch_cost_values=output.branch_cost_values.detach(),
        branch_reward_returns=output.branch_reward_values.detach(),
        branch_cost_returns=output.branch_cost_values.detach(),
        branch_reward_advantages=branch_advantages,
        branch_cost_advantages=branch_zeros.clone(),
        info_summary={},
    )


@pytest.mark.parametrize(
    ("architecture", "head_attribute"),
    [
        ("simplex_autoregressive_gaussian", "autoregressive_branch_mean_heads"),
        ("simplex_autoregressive_dirichlet", "autoregressive_dirichlet_heads"),
    ],
)
def test_standalone_optimizer_step_updates_active_actor_heads_only(
    architecture: str,
    head_attribute: str,
) -> None:
    torch.manual_seed(17)
    config = NetworkConfig(
        policy_architecture=architecture,
        branch_credit_mode="standalone",
        hidden_sizes=[16],
        equal_weight_policy_init=True,
        init_log_std=-1.5,
        dirichlet_min_concentration=0.5,
        dirichlet_init_concentration=1.5,
        dirichlet_max_concentration=8.0,
    )
    model = ActorCritic(
        obs_dim=5,
        action_dim=10,
        config=config,
        branch_sizes=[1, 3, 2, 4],
        branch_train_mask=[False, True, True, True],
    )
    observations = torch.randn((32, 5))
    batch = _standalone_policy_test_batch(model, observations)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimization = SimpleNamespace(
        epochs=1,
        minibatch_size=32,
        clip_epsilon=0.2,
        target_kl=None,
        reward_value_coef=0.0,
        cost_value_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=100.0,
    )
    heads = getattr(model, head_attribute)
    before = [
        [parameter.detach().clone() for parameter in head.parameters()]
        for head in heads
    ]

    metrics = update_ppo_actor_critic(model, optimizer, batch, optimization)

    assert metrics["optimizer_steps_completed"] == 1
    assert metrics["actor_gradient_norm_branch_1"] is None
    assert metrics["approx_kl_branch_1"] is None
    assert metrics["entropy_branch_1"] is None
    for branch_index in (1, 2, 3):
        number = branch_index + 1
        assert metrics[f"actor_gradient_norm_branch_{number}"] > 0.0
        assert metrics[f"approx_kl_branch_{number}"] is not None
        assert metrics[f"entropy_branch_{number}"] is not None
        assert torch.isfinite(torch.tensor(metrics[f"approx_kl_branch_{number}"]))
        assert torch.isfinite(torch.tensor(metrics[f"entropy_branch_{number}"]))
        assert any(
            not torch.equal(previous, current.detach())
            for previous, current in zip(
                before[branch_index],
                heads[branch_index].parameters(),
                strict=True,
            )
        )
    for previous, current in zip(before[0], heads[0].parameters(), strict=True):
        torch.testing.assert_close(current.detach(), previous)


def test_rcpo_constraint_flags_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo"])
    with pytest.raises(SystemExit):
        train.parse_args()

    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo", "--constraint-drawdown"])
    args = train.parse_args()
    assert args.constraint_drawdown
    assert not args.constraint_allocation
    assert not args.constraint_allocation_drawdown

    monkeypatch.setattr(sys, "argv", ["train.py", "--algo", "rcpo", "--constraint-allocation"])
    args = train.parse_args()
    assert args.constraint_allocation
    assert not args.constraint_drawdown
    assert not args.constraint_allocation_drawdown

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "rcpo", "--constraint-allocation-drawdown"],
    )
    args = train.parse_args()
    assert args.constraint_allocation_drawdown
    assert not args.constraint_drawdown
    assert not args.constraint_allocation

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--algo",
            "rcpo",
            "--constraint-allocation",
            "--constraint-allocation-drawdown",
        ],
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

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algo", "ppo_unconstrained", "--constraint-allocation-drawdown"],
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


def test_combined_constraint_mode_requires_fixed_alpha() -> None:
    config = ProjectConfig()
    config.rcpo.constraint_mode = "allocation_drawdown"
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
    assert config.rcpo.alpha == pytest.approx(0.0001)
    assert config.environment.allocation_constraint_cost_scale == pytest.approx(20.0)
    assert config.optimization.learning_rate == pytest.approx(0.0001)
    assert config.optimization.epochs == 3
    assert config.evaluation.validation_branch_count == 10


def test_allocation_drawdown_penalty_config_loads() -> None:
    config = load_config("configs/rcpo_allocation_drawdown_penalty.yaml")
    sync_rcpo_constraint_settings(config)

    assert config.environment.action_mode == "softmax"
    assert config.rcpo.constraint_mode == "allocation_drawdown"
    assert config.environment.constraint_mode == "allocation_drawdown"
    assert config.rcpo.alpha == pytest.approx(0.00015)
    assert config.environment.combined_drawdown_cost_weight == pytest.approx(0.25)
    assert config.rcpo.lambda_lr_up == pytest.approx(0.0005)


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
