from __future__ import annotations

import torch

from rcpo_portfolio.config import NetworkConfig, ProjectConfig, sync_rcpo_constraint_settings
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


def test_autoregressive_dirichlet_alias_resolves_to_gaussian_logits() -> None:
    config = ProjectConfig()
    config.environment.action_mode = "simplex_decomposition"
    config.network.policy_architecture = "simplex_autoregressive_dirichlet"

    sync_rcpo_constraint_settings(config)

    assert config.network.policy_architecture == "simplex_autoregressive_gaussian"
    assert config.environment.simplex_action_format == "branch_logits"
