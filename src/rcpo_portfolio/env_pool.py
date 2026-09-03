from __future__ import annotations

from typing import Any

import numpy as np

from .env import PortfolioEnv


class PortfolioEnvPool:
    """Sample one training market on each reset while exposing a Gym-like env API."""

    def __init__(self, envs: list[PortfolioEnv], seed: int | None = None) -> None:
        if not envs:
            raise ValueError("PortfolioEnvPool requires at least one environment.")
        self.envs = envs
        self.rng = np.random.default_rng(seed)
        self.active_env = envs[0]
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.num_assets = envs[0].num_assets
        self.num_risky_assets = envs[0].num_risky_assets

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = dict(options or {})
        env_index = int(options.pop("env_index", self.rng.integers(0, len(self.envs))))
        self.active_env = self.envs[env_index]
        observation, info = self.active_env.reset(options=options)
        info["env_index"] = env_index
        return observation, info

    def step(self, action):
        return self.active_env.step(action)

    def available_start_indices(self):
        return self.active_env.available_start_indices()

    def resolved_constraint_preset(self) -> dict[str, float]:
        return self.envs[0].resolved_constraint_preset()

    def simplex_branch_sizes(self) -> list[int]:
        return self.envs[0].simplex_branch_sizes()

    def simplex_branch_train_mask(self) -> list[bool]:
        return self.envs[0].simplex_branch_train_mask()


    def neutral_action(self):
        return self.envs[0].neutral_action()

    def constrained_neutral_action(self):
        return self.envs[0].constrained_neutral_action()

    @property
    def counterfactual_critic_context_dim(self) -> int:
        return self.active_env.counterfactual_critic_context_dim

    def counterfactual_critic_context(self):
        return self.active_env.counterfactual_critic_context()
