from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rcpo_portfolio.algorithms import update_rcpo_actor_critic
from rcpo_portfolio.config import load_config
from rcpo_portfolio.trainer import RCPOTrainer


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "paper_gaussian"


@pytest.mark.parametrize("arm", ["shared", "standalone", "cf_r", "cf_c", "cf_rc"])
def test_matched_credit_rollout_and_update(tmp_path, arm):
    config = load_config(CONFIG_DIR / f"{arm}.yaml")
    config.market.train_market_count = 1
    config.market.train_steps = 42
    config.market.validation_steps = 8
    config.market.test_steps = 8
    config.environment.episode_length = 8
    config.evaluation.fixed_anchor_steps = 42
    config.evaluation.validation_branch_count = 1
    config.evaluation.test_branch_count = 1
    config.optimization.total_updates = 1
    config.optimization.rollout_steps = 8
    config.optimization.minibatch_size = 8
    config.optimization.epochs = 1
    trainer = RCPOTrainer(config, "rcpo", tmp_path, 0, disable_artifacts=True)
    batch = trainer._collect_rollout()
    assert trainer.train_env.active_env.state.standalone is None
    assert len(trainer.model.branch_reward_values) == len(trainer.model.branch_cost_values) == 4
    assert trainer.matched_credit_protocol["outer_branch_weight"] == "active_mask"
    if arm == "shared":
        for field in ("reward_advantages", "cost_advantages"):
            common = getattr(batch, field)
            branch = getattr(batch, f"branch_{field}")
            torch.testing.assert_close(branch, common[:, None].expand_as(branch))
    metrics, _, _ = update_rcpo_actor_critic(
        trainer.model, trainer.optimizer, batch, config.optimization,
        lambda_value=0.1, alpha=0.0, lambda_lr=0.001,
    )
    assert metrics["optimizer_steps_completed"] == 1
    assert metrics["lambda_update_count"] == 1
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
