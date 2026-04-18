from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvironmentConfig, MarketConfig
from .market import MarketSlice


@dataclass
class EpisodeState:
    start_index: int
    current_index: int
    steps_elapsed: int
    weights: np.ndarray
    previous_turnover: float
    net_returns: list[float]


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvironmentConfig,
        market: MarketSlice,
        market_config: MarketConfig,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.market = market
        self.market_config = market_config
        self.num_risky_assets = self.market.risky_returns.shape[1]
        self.num_assets = self.num_risky_assets + 1
        self.lookback = market_config.lookback
        self.transaction_cost_rate = config.transaction_cost_bps / 10_000.0
        self.rng = np.random.default_rng(seed)
        self.state: EpisodeState | None = None
        self._valid_start_indices = self._compute_start_indices()
        self._validate_constraint_settings()
        self._resolved_constraint_preset = self._resolve_constraint_preset()

        obs_dim = (
            self.lookback * self.num_risky_assets
            + self.num_risky_assets
            + self.num_risky_assets
            + self.num_assets
            + 6
        )
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-bound,
            high=bound,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-bound,
            high=bound,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

    def _compute_start_indices(self) -> np.ndarray:
        last_start = len(self.market.risky_returns) - self.config.episode_length
        if last_start < self.lookback:
            raise ValueError("Market slice is too short for the configured episode length.")
        return np.arange(self.lookback, last_start + 1, dtype=np.int64)

    def _validate_constraint_settings(self) -> None:
        if self.config.constraint_mode not in {"downside", "sortino"}:
            raise ValueError(
                "Environment constraint_mode must be either 'downside' or 'sortino'."
            )
        if self.config.downside_cost_scale <= 0.0:
            raise ValueError("downside_cost_scale must be positive.")
        if self.config.sortino_window <= 0:
            raise ValueError("sortino_window must be positive.")
        if self.config.sortino_min_periods <= 0:
            raise ValueError("sortino_min_periods must be positive.")
        if self.config.sortino_cost_scale <= 0.0:
            raise ValueError("sortino_cost_scale must be positive.")

    def available_start_indices(self) -> np.ndarray:
        return self._valid_start_indices.copy()

    def _resolve_constraint_preset(self) -> dict[str, float]:
        preset_name = self.config.active_constraint_preset
        if preset_name not in self.config.constraint_presets:
            raise ValueError(f"Unknown constraint preset: {preset_name}")
        resolved = dict(self.config.constraint_presets[preset_name])
        resolved["preset_name"] = preset_name
        return resolved

    def resolved_constraint_preset(self) -> dict[str, float]:
        return dict(self._resolved_constraint_preset)

    def _weights_from_action(self, action: np.ndarray) -> np.ndarray:
        logits = np.asarray(action, dtype=np.float32)
        if logits.shape != (self.num_assets,):
            raise ValueError(
                f"Expected action shape {(self.num_assets,)}, got {logits.shape}."
            )
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.sum(weights)
        return weights.astype(np.float32)

    def _group_weights(self, weights: np.ndarray) -> dict[str, float]:
        risky_weights = weights[1:]
        group_a_weight = float(np.sum(risky_weights[self.config.group_a_indices]))
        group_b_weight = float(np.sum(risky_weights[self.config.group_b_indices]))
        return {
            "group_a_weight": group_a_weight,
            "group_b_weight": group_b_weight,
        }

    def _constraint_components(self, weights: np.ndarray) -> dict[str, float]:
        group_weights = self._group_weights(weights)
        group_a_min_weight = float(self._resolved_constraint_preset["group_a_min_weight"])
        group_b_max_weight = float(self._resolved_constraint_preset["group_b_max_weight"])
        group_a_min_violation = float(
            (
                max(group_a_min_weight - group_weights["group_a_weight"], 0.0)
                / max(group_a_min_weight, 1e-6)
            )
            ** 2
        )
        group_b_max_violation = float(
            (
                max(group_weights["group_b_weight"] - group_b_max_weight, 0.0)
                / max(group_b_max_weight, 1e-6)
            )
            ** 2
        )
        return {
            **group_weights,
            "group_a_min_violation_cost": group_a_min_violation,
            "group_b_max_violation_cost": group_b_max_violation,
            "group_a_min_weight": group_a_min_weight,
            "group_b_max_weight": group_b_max_weight,
        }

    def _sortino_components(self, net_simple_return: float) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("Environment has not been reset.")
        episode_returns = [*self.state.net_returns, float(net_simple_return)]
        window = min(int(self.config.sortino_window), len(episode_returns))
        trailing_returns = np.asarray(episode_returns[-window:], dtype=np.float32)
        if len(trailing_returns) < int(self.config.sortino_min_periods):
            return {
                "sortino_ratio": 0.0,
                "sortino_violation_cost": 0.0,
            }

        mean_return = float(np.mean(trailing_returns))
        downside_deviation = float(
            np.sqrt(np.mean(np.square(np.minimum(trailing_returns, 0.0))))
        )
        if downside_deviation <= 1e-12:
            sortino_ratio = float(self.config.sortino_target) if mean_return > 0.0 else 0.0
        else:
            sortino_ratio = float(np.sqrt(252.0) * mean_return / downside_deviation)
        violation = max(0.0, float(self.config.sortino_target) - sortino_ratio)
        return {
            "sortino_ratio": sortino_ratio,
            "sortino_violation_cost": float(
                violation**2 / max(float(self.config.sortino_cost_scale), 1e-12)
            ),
        }

    def _get_observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment has not been reset.")
        current_index = self.state.current_index
        trailing_returns = self.market.risky_returns[current_index - self.lookback : current_index]
        rolling_mean = trailing_returns.mean(axis=0)
        rolling_vol = trailing_returns.std(axis=0)
        group_components = self._constraint_components(self.state.weights)
        turnover_cap_slack = self.config.turnover_cap - self.state.previous_turnover
        observation = np.concatenate(
            [
                trailing_returns.reshape(-1),
                rolling_mean,
                rolling_vol,
                self.state.weights,
                np.asarray([self.state.previous_turnover], dtype=np.float32),
                np.asarray([group_components["group_a_weight"]], dtype=np.float32),
                np.asarray([group_components["group_b_weight"]], dtype=np.float32),
                np.asarray(
                    [
                        group_components["group_a_min_weight"]
                        - group_components["group_a_weight"]
                    ],
                    dtype=np.float32,
                ),
                np.asarray(
                    [
                        group_components["group_b_weight"]
                        - group_components["group_b_max_weight"]
                    ],
                    dtype=np.float32,
                ),
                np.asarray([turnover_cap_slack], dtype=np.float32),
            ]
        )
        return observation.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if "start_index" in options:
            start_index = int(options["start_index"])
        else:
            start_index = int(self.rng.choice(self._valid_start_indices))
        if start_index not in set(self._valid_start_indices.tolist()):
            raise ValueError(f"Invalid start index {start_index}.")
        initial_weights = np.zeros(self.num_assets, dtype=np.float32)
        initial_weights[0] = 1.0
        self.state = EpisodeState(
            start_index=start_index,
            current_index=start_index,
            steps_elapsed=0,
            weights=initial_weights,
            previous_turnover=0.0,
            net_returns=[],
        )
        return self._get_observation(), {"start_index": start_index}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        weights = self._weights_from_action(action)
        current_returns = self.market.risky_returns[self.state.current_index]
        raw_simple_return = float(np.dot(weights[1:], current_returns.astype(np.float32)))
        turnover = float(np.sum(np.abs(weights - self.state.weights)))
        transaction_cost = float(self.transaction_cost_rate * turnover)
        net_simple_return = raw_simple_return - transaction_cost
        reward = float(np.log1p(np.clip(net_simple_return, -0.999999, None)))
        downside_cost = float(max(0.0, -net_simple_return) ** 2)
        constraint_components = self._constraint_components(weights)
        sortino_components = self._sortino_components(net_simple_return)
        normalized_downside_cost = downside_cost / float(self.config.downside_cost_scale)
        if self.config.constraint_mode == "downside":
            constraint_cost = float(normalized_downside_cost)
        else:
            constraint_cost = float(sortino_components["sortino_violation_cost"])

        self.state.weights = weights
        self.state.previous_turnover = turnover
        self.state.net_returns.append(net_simple_return)
        self.state.current_index += 1
        self.state.steps_elapsed += 1

        terminated = self.state.steps_elapsed >= self.config.episode_length
        truncated = False
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated or truncated
            else self._get_observation()
        )
        info = {
            "raw_return": raw_simple_return,
            "net_return": net_simple_return,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "downside_cost": downside_cost,
            "normalized_downside_cost": normalized_downside_cost,
            "constraint_cost": constraint_cost,
            "constraint_mode": self.config.constraint_mode,
            **sortino_components,
            "weights": weights.copy(),
            "regime": int(self.market.regimes[self.state.current_index - 1]),
            **constraint_components,
        }
        return observation, reward, terminated, truncated, info
