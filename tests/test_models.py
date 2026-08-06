from __future__ import annotations

import torch

from rcpo_portfolio.config import (
    NetworkConfig,
    ProjectConfig,
    load_config,
    sync_rcpo_constraint_settings,
)
from rcpo_portfolio.models import ActorCritic


def test_flat_gaussian_policy_preserves_action_contract() -> None:
    config = NetworkConfig(policy_architecture="flat_gaussian", hidden_sizes=[8])
    model = ActorCritic(obs_dim=4, action_dim=6, config=config)

    action, log_prob, entropy, reward_value, cost_value = model.get_action_and_value(
        torch.zeros((3, 4)),
    )

    assert action.shape == (3, 6)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)
    assert reward_value.shape == (3,)
    assert cost_value.shape == (3,)


def test_default_config_exposes_standalone_credit_and_dirichlet_settings() -> None:
    config = load_config("configs/default.yaml")

    assert config.network.policy_architecture == "simplex_autoregressive_dirichlet"
    assert config.network.branch_credit_mode == "standalone"
    assert config.environment.initial_portfolio_mode == "constrained_neutral"
    assert config.environment.simplex_action_format == "branch_weights"
    assert config.network.dirichlet_min_concentration == 0.5
    assert config.network.dirichlet_init_concentration == 1.5
    assert config.network.dirichlet_max_concentration == 8.0


def test_parallel_branch_gaussian_uses_four_branch_heads() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_branch_gaussian",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=13,
        config=config,
        branch_sizes=[1, 3, 3, 6],
    )

    action, log_prob, entropy, _, _ = model.get_action_and_value(torch.zeros((2, 4)))

    assert len(model.branch_mean_heads) == 4
    assert [head.out_features for head in model.branch_mean_heads] == [1, 3, 3, 6]
    assert action.shape == (2, 13)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)


def test_autoregressive_gaussian_outputs_branch_logits() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_gaussian",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=13,
        config=config,
        branch_sizes=[1, 3, 3, 6],
    )

    action, log_prob, entropy, _, _ = model.get_action_and_value(torch.zeros((5, 4)))

    assert action.shape == (5, 13)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(action).all()
    torch.testing.assert_close(action[:, 0], torch.zeros(5))


def test_standalone_credit_builds_four_reward_and_cost_critics() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_gaussian",
        branch_credit_mode="standalone",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=12,
        config=config,
        branch_sizes=[1, 3, 2, 6],
    )

    output = model.get_policy_output(torch.zeros((3, 4)))

    assert len(model.branch_reward_values) == 4
    assert len(model.branch_cost_values) == 4
    assert output.branch_reward_values.shape == (3, 4)
    assert output.branch_cost_values.shape == (3, 4)


def test_autoregressive_gaussian_deterministic_action_is_zero_at_init() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_gaussian",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=13,
        config=config,
        branch_sizes=[1, 3, 3, 6],
    )

    action, log_prob, entropy, _, _ = model.get_action_and_value(
        torch.zeros((1, 4)),
        deterministic=True,
    )

    torch.testing.assert_close(action, torch.zeros((1, 13)))
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_autoregressive_gaussian_conditions_on_previous_branch_softmax_weights() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_gaussian",
        hidden_sizes=[],
        equal_weight_policy_init=True,
        init_log_std=-1.0,
    )
    model = ActorCritic(
        obs_dim=3,
        action_dim=8,
        config=config,
        branch_sizes=[2, 2, 2, 2],
    )

    with torch.no_grad():
        branch_2_head = model.autoregressive_branch_mean_heads[1]
        branch_2_head.weight.zero_()
        branch_2_head.bias.zero_()
        branch_2_head.weight[0, 3] = 4.0

    obs = torch.zeros((1, 3))
    branch_1_left = torch.tensor([[5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    branch_1_right = torch.tensor([[-5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    _, left_log_prob, _, _, _ = model.get_action_and_value(obs, action=branch_1_left)
    _, right_log_prob, _, _, _ = model.get_action_and_value(obs, action=branch_1_right)

    assert not torch.allclose(left_log_prob, right_log_prob)


def test_autoregressive_dirichlet_resolves_to_branch_weights() -> None:
    config = ProjectConfig()
    config.environment.action_mode = "simplex_decomposition"
    config.network.policy_architecture = "simplex_autoregressive_dirichlet"

    sync_rcpo_constraint_settings(config)

    assert config.network.policy_architecture == "simplex_autoregressive_dirichlet"
    assert config.environment.simplex_action_format == "branch_weights"


def test_autoregressive_dirichlet_samples_valid_bounded_branch_weights() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_dirichlet",
        hidden_sizes=[8],
        dirichlet_min_concentration=0.5,
        dirichlet_init_concentration=12.0,
        dirichlet_max_concentration=80.0,
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=12,
        config=config,
        branch_sizes=[1, 3, 2, 6],
    )

    output = model.get_policy_output(torch.zeros((5, 4)))
    branches = torch.split(output.action, [1, 3, 2, 6], dim=-1)

    for branch in branches:
        assert torch.all(branch > 0.0)
        torch.testing.assert_close(branch.sum(dim=-1), torch.ones(5))
    assert torch.isfinite(output.log_prob).all()
    assert torch.isfinite(output.entropy).all()
    for head in model.autoregressive_dirichlet_heads:
        raw = head.bias
        concentration = 0.5 + (80.0 - 0.5) * torch.sigmoid(raw)
        torch.testing.assert_close(concentration, torch.full_like(concentration, 12.0))


def test_autoregressive_dirichlet_deterministic_initial_action_is_uniform() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_dirichlet",
        hidden_sizes=[8],
        dirichlet_min_concentration=0.5,
        dirichlet_init_concentration=12.0,
        dirichlet_max_concentration=80.0,
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=8,
        config=config,
        branch_sizes=[1, 2, 2, 3],
    )

    output = model.get_policy_output(torch.zeros((1, 4)), deterministic=True)
    expected = torch.tensor([[1.0, 0.5, 0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]])
    torch.testing.assert_close(output.action, expected)
    torch.testing.assert_close(output.branch_log_probs[:, 0], torch.zeros(1))
    torch.testing.assert_close(output.branch_entropies[:, 0], torch.zeros(1))

def test_inactive_parallel_gaussian_branch_is_fixed_and_has_no_gradient() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_branch_gaussian",
        branch_credit_mode="standalone",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=8,
        config=config,
        branch_sizes=[2, 2, 2, 2],
        branch_train_mask=[False, True, True, True],
    )

    output = model.get_policy_output(torch.zeros((3, 4)))
    torch.testing.assert_close(output.action[:, :2], torch.zeros((3, 2)))
    torch.testing.assert_close(output.branch_log_probs[:, 0], torch.zeros(3))
    torch.testing.assert_close(output.branch_entropies[:, 0], torch.zeros(3))
    torch.testing.assert_close(output.branch_reward_values[:, 0], torch.zeros(3))
    torch.testing.assert_close(output.branch_cost_values[:, 0], torch.zeros(3))

    loss = (
        output.branch_log_probs[:, 1:].sum()
        + output.branch_reward_values[:, 1:].sum()
        + output.branch_cost_values[:, 1:].sum()
    )
    loss.backward()

    assert model.branch_mean_heads[0].weight.grad is None
    assert model.branch_log_stds[0].grad is None
    assert model.branch_reward_values[0].weight.grad is None
    assert model.branch_cost_values[0].weight.grad is None
    assert model.branch_mean_heads[1].weight.grad is not None


def test_inactive_autoregressive_gaussian_branch_is_neutral() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_gaussian",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=8,
        config=config,
        branch_sizes=[2, 2, 2, 2],
        branch_train_mask=[False, True, True, True],
    )

    output = model.get_policy_output(torch.zeros((3, 4)))

    torch.testing.assert_close(output.action[:, :2], torch.zeros((3, 2)))
    torch.testing.assert_close(output.branch_log_probs[:, 0], torch.zeros(3))
    torch.testing.assert_close(output.branch_entropies[:, 0], torch.zeros(3))
    assert torch.isfinite(output.log_prob).all()


def test_inactive_autoregressive_dirichlet_branch_is_uniform() -> None:
    config = NetworkConfig(
        policy_architecture="simplex_autoregressive_dirichlet",
        hidden_sizes=[8],
        dirichlet_min_concentration=0.5,
        dirichlet_init_concentration=1.5,
        dirichlet_max_concentration=8.0,
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=8,
        config=config,
        branch_sizes=[2, 2, 2, 2],
        branch_train_mask=[False, True, True, True],
    )

    output = model.get_policy_output(torch.zeros((3, 4)))

    torch.testing.assert_close(output.action[:, :2], torch.full((3, 2), 0.5))
    torch.testing.assert_close(output.branch_log_probs[:, 0], torch.zeros(3))
    torch.testing.assert_close(output.branch_entropies[:, 0], torch.zeros(3))
    assert torch.isfinite(output.log_prob).all()


def test_flat_gaussian_masks_inactive_simplex_action_components() -> None:
    config = NetworkConfig(
        policy_architecture="flat_gaussian",
        hidden_sizes=[8],
    )
    model = ActorCritic(
        obs_dim=4,
        action_dim=8,
        config=config,
        branch_sizes=[2, 2, 2, 2],
        branch_train_mask=[False, True, True, True],
    )

    output = model.get_policy_output(torch.zeros((3, 4)))

    torch.testing.assert_close(output.action[:, :2], torch.zeros((3, 2)))
    assert torch.isfinite(output.log_prob).all()
    assert torch.isfinite(output.entropy).all()
