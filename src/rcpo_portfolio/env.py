from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvironmentConfig, MarketConfig
from .market import MarketSlice
from .simplex import SimplexDecomposition, build_simplex_decomposition


@dataclass
class EpisodeState:
    start_index: int
    current_index: int
    steps_elapsed: int
    weights: np.ndarray
    previous_turnover: float
    net_returns: list[float]
    portfolio_value: float
    running_peak_value: float
    current_drawdown: float
    max_drawdown: float
    benchmark_portfolio_value: float
    benchmark_running_peak_value: float
    benchmark_current_drawdown: float
    benchmark_max_drawdown: float
    benchmark_has_rebalanced: bool
    branch_weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    branch_portfolio_values: np.ndarray
    branch_running_peak_values: np.ndarray
    branch_current_drawdowns: np.ndarray
    branch_max_drawdowns: np.ndarray


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
        self._initial_cash_weights = np.zeros(self.num_assets, dtype=np.float32)
        self._initial_cash_weights[0] = 1.0
        self._validate_constraint_settings()
        self._resolved_constraint_preset = self._resolve_constraint_preset()
        self._simplex_decomposition: SimplexDecomposition | None = None
        if self.config.action_mode == "simplex_decomposition":
            self._simplex_decomposition = build_simplex_decomposition(
                num_assets=self.num_assets,
                constraint_1_indices=self.config.allocation_constraint_1_indices,
                constraint_2_indices=self.config.allocation_constraint_2_indices,
                constraint_1_min_weight=float(
                    self._resolved_constraint_preset[
                        "allocation_constraint_1_min_weight"
                    ]
                ),
                constraint_2_min_weight=float(
                    self._resolved_constraint_preset[
                        "allocation_constraint_2_min_weight"
                    ]
                ),
            )
        action_dim = (
            self.num_assets
            if self._simplex_decomposition is None
            else self._simplex_decomposition.action_dim
        )

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
            shape=(action_dim,),
            dtype=np.float32,
        )
        self._initial_portfolio_weights = self._resolve_initial_portfolio_weights()
        self._initial_branch_weights = self._resolve_initial_branch_weights()
        self._benchmark_mode = self.config.drawdown_benchmark_mode
        self._benchmark_weights = self._resolve_benchmark_weights()
        self._benchmark_risky_weights = self._benchmark_weights[1:]
        self._benchmark_raw_returns = (
            self.market.risky_returns @ self._benchmark_risky_weights
        ).astype(np.float32)
        self._benchmark_initial_turnover = float(
            np.sum(np.abs(self._benchmark_weights - self._initial_portfolio_weights))
        )

    def _compute_start_indices(self) -> np.ndarray:
        last_start = len(self.market.risky_returns) - self.config.episode_length
        if last_start < self.lookback:
            raise ValueError("Market slice is too short for the configured episode length.")
        return np.arange(self.lookback, last_start + 1, dtype=np.int64)

    def _validate_constraint_settings(self) -> None:
        if self.config.action_mode not in {"softmax", "simplex_decomposition"}:
            raise ValueError(
                "Environment action_mode must be either 'softmax' or "
                "'simplex_decomposition'."
            )
        if self.config.simplex_action_format not in {"branch_logits", "branch_weights"}:
            raise ValueError(
                "Environment simplex_action_format must be either 'branch_logits' "
                "or 'branch_weights'."
            )
        if self.config.initial_portfolio_mode not in {"all_cash", "constrained_neutral"}:
            raise ValueError(
                "initial_portfolio_mode must be either 'all_cash' or "
                "'constrained_neutral'."
            )
        if self.config.constraint_mode != "max_drawdown":
            raise ValueError(
                "Environment constraint_mode must be 'max_drawdown'."
            )
        if self.config.drawdown_budget_floor < 0.0:
            raise ValueError("drawdown_budget_floor cannot be negative.")
        if self.config.drawdown_benchmark_mode not in {
            "true_equal_weight",
            "constrained_neutral",
        }:
            raise ValueError(
                "drawdown_benchmark_mode must be either 'true_equal_weight' "
                "or 'constrained_neutral'."
            )
        if self.config.benchmark_drawdown_margin <= 0.0:
            raise ValueError("benchmark_drawdown_margin must be positive.")
        if self.config.drawdown_cost_scale <= 0.0:
            raise ValueError("drawdown_cost_scale must be positive.")
        if self.config.diversification_beta < 0.0:
            raise ValueError("diversification_beta cannot be negative.")

    def available_start_indices(self) -> np.ndarray:
        return self._valid_start_indices.copy()

    def _resolve_constraint_preset(self) -> dict[str, float]:
        preset_name = self.config.active_constraint_preset
        if preset_name not in self.config.constraint_presets:
            raise ValueError(f"Unknown constraint preset: {preset_name}")
        resolved = dict(self.config.constraint_presets[preset_name])
        if (
            "allocation_constraint_1_min_weight" not in resolved
            or "allocation_constraint_2_min_weight" not in resolved
        ):
            raise ValueError(
                "Constraint presets must define allocation_constraint_1_min_weight "
                "and allocation_constraint_2_min_weight."
            )
        resolved["preset_name"] = preset_name
        return resolved

    def resolved_constraint_preset(self) -> dict[str, float]:
        return dict(self._resolved_constraint_preset)

    def simplex_branch_sizes(self) -> list[int]:
        if self._simplex_decomposition is None:
            return []
        return [len(indices) for indices in self._simplex_decomposition.branch_indices]

    def neutral_action(self) -> np.ndarray:
        if self.config.action_mode == "simplex_decomposition":
            if self._simplex_decomposition is None:
                raise RuntimeError("Simplex decomposition has not been initialized.")
            if self.config.simplex_action_format == "branch_weights":
                return self._simplex_decomposition.neutral_branch_weights()
        return np.zeros(self.action_space.shape[0], dtype=np.float32)

    def benchmark_weights(self) -> np.ndarray:
        return self._benchmark_weights.copy()

    def initial_portfolio_weights(self) -> np.ndarray:
        return self._initial_portfolio_weights.copy()

    def _resolve_initial_portfolio_weights(self) -> np.ndarray:
        if self.config.initial_portfolio_mode == "all_cash":
            return self._initial_cash_weights.copy()
        if self.config.initial_portfolio_mode == "constrained_neutral":
            return self._weights_from_action(self.neutral_action()).astype(np.float32)
        raise ValueError(
            f"Unsupported initial_portfolio_mode: {self.config.initial_portfolio_mode}"
        )

    def _resolve_initial_branch_weights(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._simplex_decomposition is None:
            empty = np.zeros(self.num_assets, dtype=np.float32)
            return (empty.copy(), empty.copy(), empty.copy(), empty.copy())
        return self._simplex_decomposition.neutral_padded_branches()

    def _resolve_benchmark_weights(self) -> np.ndarray:
        if self.config.drawdown_benchmark_mode == "true_equal_weight":
            return np.full(
                self.num_assets,
                1.0 / float(self.num_assets),
                dtype=np.float32,
            )
        if self.config.drawdown_benchmark_mode == "constrained_neutral":
            return self._weights_from_action(self.neutral_action()).astype(np.float32)
        raise ValueError(f"Unsupported drawdown_benchmark_mode: {self.config.drawdown_benchmark_mode}")

    def _weights_from_action(self, action: np.ndarray) -> np.ndarray:
        weights, _ = self._weights_and_action_components(action)
        return weights

    def _weights_and_action_components(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.config.action_mode == "simplex_decomposition":
            if self._simplex_decomposition is None:
                raise RuntimeError("Simplex decomposition has not been initialized.")
            if self.config.simplex_action_format == "branch_weights":
                result = self._simplex_decomposition.map_branch_weights(action)
            else:
                result = self._simplex_decomposition.map_logits(action)
            return result.weights, {
                **result.diagnostics,
                "simplex_branch_weights": tuple(
                    branch.copy() for branch in result.branch_weights
                ),
            }

        logits = np.asarray(action, dtype=np.float32)
        if logits.shape != (self.num_assets,):
            raise ValueError(
                f"Expected action shape {(self.num_assets,)}, got {logits.shape}."
            )
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.sum(weights)
        return weights.astype(np.float32), {
            "simplex_z1": 0.0,
            "simplex_z2": 0.0,
            "simplex_z3": 0.0,
            "simplex_z4": 0.0,
            "simplex_z2_intersection": 0.0,
            "simplex_branch_weights": tuple(
                np.zeros(self.num_assets, dtype=np.float32) for _ in range(4)
            ),
        }

    def _allocation_constraint_weights(self, weights: np.ndarray) -> dict[str, float]:
        constraint_1_weight = float(
            np.sum(weights[self.config.allocation_constraint_1_indices])
        )
        constraint_2_weight = float(
            np.sum(weights[self.config.allocation_constraint_2_indices])
        )
        return {
            "allocation_constraint_1_weight": constraint_1_weight,
            "allocation_constraint_2_weight": constraint_2_weight,
        }

    def _constraint_components(self, weights: np.ndarray) -> dict[str, float]:
        allocation_weights = self._allocation_constraint_weights(weights)
        constraint_1_min_weight = float(
            self._resolved_constraint_preset["allocation_constraint_1_min_weight"]
        )
        constraint_2_min_weight = float(
            self._resolved_constraint_preset["allocation_constraint_2_min_weight"]
        )
        concentration = float(np.sum(np.square(weights)))
        equal_weight_concentration = 1.0 / float(self.num_assets)
        excess_concentration = max(0.0, concentration - equal_weight_concentration)
        if excess_concentration < 1e-7:
            excess_concentration = 0.0
        excess_concentration_cost = float(
            excess_concentration / max(1.0 - equal_weight_concentration, 1e-12)
        )
        diversification_cost = float(
            self.config.diversification_beta * excess_concentration_cost
        )
        constraint_1_violation = float(
            (
                max(
                    constraint_1_min_weight
                    - allocation_weights["allocation_constraint_1_weight"],
                    0.0,
                )
                / max(constraint_1_min_weight, 1e-6)
            )
            ** 2
        )
        constraint_2_violation = float(
            (
                max(
                    constraint_2_min_weight
                    - allocation_weights["allocation_constraint_2_weight"],
                    0.0,
                )
                / max(constraint_2_min_weight, 1e-6)
            )
            ** 2
        )
        return {
            **allocation_weights,
            "allocation_constraint_1_violation_cost": constraint_1_violation,
            "allocation_constraint_2_violation_cost": constraint_2_violation,
            "allocation_constraint_1_min_weight": constraint_1_min_weight,
            "allocation_constraint_2_min_weight": constraint_2_min_weight,
            "concentration": concentration,
            "excess_concentration_cost": excess_concentration_cost,
            "diversification_cost": diversification_cost,
        }

    @staticmethod
    def _updated_drawdown_state(
        portfolio_value: float,
        running_peak_value: float,
        max_drawdown: float,
        net_simple_return: float,
    ) -> dict[str, float]:
        growth = max(1e-12, 1.0 + float(net_simple_return))
        next_portfolio_value = float(portfolio_value * growth)
        next_running_peak_value = float(max(running_peak_value, next_portfolio_value))
        current_drawdown = float(
            (next_running_peak_value - next_portfolio_value)
            / max(next_running_peak_value, 1e-12)
        )
        next_max_drawdown = float(max(max_drawdown, current_drawdown))
        return {
            "portfolio_value": next_portfolio_value,
            "running_peak_value": next_running_peak_value,
            "current_drawdown": current_drawdown,
            "max_drawdown": next_max_drawdown,
        }

    def _benchmark_drawdown_components(self) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("Environment has not been reset.")
        raw_simple_return = float(self._benchmark_raw_returns[self.state.current_index])
        turnover = (
            0.0
            if self.state.benchmark_has_rebalanced
            else self._benchmark_initial_turnover
        )
        transaction_cost = float(self.transaction_cost_rate * turnover)
        net_simple_return = raw_simple_return - transaction_cost
        drawdown_state = self._updated_drawdown_state(
            portfolio_value=self.state.benchmark_portfolio_value,
            running_peak_value=self.state.benchmark_running_peak_value,
            max_drawdown=self.state.benchmark_max_drawdown,
            net_simple_return=net_simple_return,
        )
        effective_drawdown_budget = float(
            max(
                float(self.config.drawdown_budget_floor),
                float(self.config.benchmark_drawdown_margin)
                * float(drawdown_state["max_drawdown"]),
            )
        )
        return {
            "benchmark_raw_return": raw_simple_return,
            "drawdown_benchmark_mode": self._benchmark_mode,
            "benchmark_turnover": float(turnover),
            "benchmark_transaction_cost": transaction_cost,
            "benchmark_net_return": net_simple_return,
            "benchmark_portfolio_value": float(drawdown_state["portfolio_value"]),
            "benchmark_running_peak_value": float(drawdown_state["running_peak_value"]),
            "benchmark_current_drawdown": float(drawdown_state["current_drawdown"]),
            "benchmark_max_drawdown": float(drawdown_state["max_drawdown"]),
            "benchmark_weights": self._benchmark_weights.copy(),
            "effective_drawdown_budget": effective_drawdown_budget,
        }

    def _drawdown_components(
        self,
        net_simple_return: float,
        effective_drawdown_budget: float,
    ) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("Environment has not been reset.")
        drawdown_state = self._updated_drawdown_state(
            portfolio_value=self.state.portfolio_value,
            running_peak_value=self.state.running_peak_value,
            max_drawdown=self.state.max_drawdown,
            net_simple_return=net_simple_return,
        )
        drawdown_gap = float(drawdown_state["max_drawdown"] - effective_drawdown_budget)
        drawdown_violation = float(
            max(0.0, drawdown_gap)
        )
        drawdown_constraint_cost = float(
            drawdown_violation**2 / max(float(self.config.drawdown_cost_scale), 1e-12)
        )
        return {
            "portfolio_value": float(drawdown_state["portfolio_value"]),
            "running_peak_value": float(drawdown_state["running_peak_value"]),
            "current_drawdown": float(drawdown_state["current_drawdown"]),
            "max_drawdown": float(drawdown_state["max_drawdown"]),
            "drawdown_gap": drawdown_gap,
            "drawdown_violation": drawdown_violation,
            "drawdown_constraint_cost": drawdown_constraint_cost,
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
                np.asarray(
                    [group_components["allocation_constraint_1_weight"]],
                    dtype=np.float32,
                ),
                np.asarray(
                    [group_components["allocation_constraint_2_weight"]],
                    dtype=np.float32,
                ),
                np.asarray(
                    [
                        group_components["allocation_constraint_1_min_weight"]
                        - group_components["allocation_constraint_1_weight"]
                    ],
                    dtype=np.float32,
                ),
                np.asarray(
                    [
                        group_components["allocation_constraint_2_min_weight"]
                        - group_components["allocation_constraint_2_weight"]
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
        initial_weights = self._initial_portfolio_weights.copy()
        initial_branch_weights = tuple(
            weights.copy() for weights in self._initial_branch_weights
        )
        self.state = EpisodeState(
            start_index=start_index,
            current_index=start_index,
            steps_elapsed=0,
            weights=initial_weights,
            previous_turnover=0.0,
            net_returns=[],
            portfolio_value=1.0,
            running_peak_value=1.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            benchmark_portfolio_value=1.0,
            benchmark_running_peak_value=1.0,
            benchmark_current_drawdown=0.0,
            benchmark_max_drawdown=0.0,
            benchmark_has_rebalanced=bool(
                np.allclose(
                    self._benchmark_weights,
                    self._initial_portfolio_weights,
                    atol=1e-8,
                )
            ),
            branch_weights=initial_branch_weights,
            branch_portfolio_values=np.ones(4, dtype=np.float64),
            branch_running_peak_values=np.ones(4, dtype=np.float64),
            branch_current_drawdowns=np.zeros(4, dtype=np.float64),
            branch_max_drawdowns=np.zeros(4, dtype=np.float64),
        )
        return self._get_observation(), {"start_index": start_index}

    def _standalone_branch_components(
        self,
        branch_weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        current_returns: np.ndarray,
        effective_drawdown_budget: float,
        z_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Environment has not been reset.")
        rewards = np.zeros(4, dtype=np.float32)
        costs = np.zeros(4, dtype=np.float32)
        turnovers = np.zeros(4, dtype=np.float32)
        net_returns = np.zeros(4, dtype=np.float32)
        max_drawdowns = np.zeros(4, dtype=np.float32)
        for branch_index, weights in enumerate(branch_weights):
            previous_weights = self.state.branch_weights[branch_index]
            turnover = float(np.sum(np.abs(weights - previous_weights)))
            raw_return = float(np.dot(weights[1:], current_returns.astype(np.float32)))
            net_return = raw_return - self.transaction_cost_rate * turnover
            drawdown_state = self._updated_drawdown_state(
                portfolio_value=float(self.state.branch_portfolio_values[branch_index]),
                running_peak_value=float(
                    self.state.branch_running_peak_values[branch_index]
                ),
                max_drawdown=float(self.state.branch_max_drawdowns[branch_index]),
                net_simple_return=net_return,
            )
            violation = max(
                0.0,
                float(drawdown_state["max_drawdown"]) - effective_drawdown_budget,
            )
            cost = violation**2 / max(float(self.config.drawdown_cost_scale), 1e-12)
            rewards[branch_index] = float(
                np.log1p(np.clip(net_return, -0.999999, None))
            )
            costs[branch_index] = float(cost)
            turnovers[branch_index] = float(turnover)
            net_returns[branch_index] = float(net_return)
            max_drawdowns[branch_index] = float(drawdown_state["max_drawdown"])
            self.state.branch_portfolio_values[branch_index] = float(
                drawdown_state["portfolio_value"]
            )
            self.state.branch_running_peak_values[branch_index] = float(
                drawdown_state["running_peak_value"]
            )
            self.state.branch_current_drawdowns[branch_index] = float(
                drawdown_state["current_drawdown"]
            )
            self.state.branch_max_drawdowns[branch_index] = float(
                drawdown_state["max_drawdown"]
            )
        self.state.branch_weights = tuple(weights.copy() for weights in branch_weights)
        return {
            "branch_rewards": rewards,
            "branch_costs": costs,
            "branch_turnovers": turnovers,
            "branch_transaction_costs": turnovers * self.transaction_cost_rate,
            "branch_net_returns": net_returns,
            "branch_max_drawdowns": max_drawdowns,
            "branch_z_values": z_values.astype(np.float32),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        weights, action_components = self._weights_and_action_components(action)
        current_returns = self.market.risky_returns[self.state.current_index]
        raw_simple_return = float(np.dot(weights[1:], current_returns.astype(np.float32)))
        turnover = float(np.sum(np.abs(weights - self.state.weights)))
        transaction_cost = float(self.transaction_cost_rate * turnover)
        net_simple_return = raw_simple_return - transaction_cost
        reward = float(np.log1p(np.clip(net_simple_return, -0.999999, None)))
        constraint_components = self._constraint_components(weights)
        benchmark_components = self._benchmark_drawdown_components()
        drawdown_components = self._drawdown_components(
            net_simple_return,
            benchmark_components["effective_drawdown_budget"],
        )
        constraint_cost = float(drawdown_components["drawdown_constraint_cost"])
        branch_weights = action_components["simplex_branch_weights"]
        z_values = np.asarray(
            [
                action_components["simplex_z1"],
                action_components["simplex_z2"],
                action_components["simplex_z3"],
                action_components["simplex_z4"],
            ],
            dtype=np.float32,
        )
        branch_components = self._standalone_branch_components(
            branch_weights,
            current_returns,
            float(benchmark_components["effective_drawdown_budget"]),
            z_values,
        )

        self.state.weights = weights
        self.state.previous_turnover = turnover
        self.state.net_returns.append(net_simple_return)
        self.state.portfolio_value = float(drawdown_components["portfolio_value"])
        self.state.running_peak_value = float(drawdown_components["running_peak_value"])
        self.state.current_drawdown = float(drawdown_components["current_drawdown"])
        self.state.max_drawdown = float(drawdown_components["max_drawdown"])
        self.state.benchmark_portfolio_value = float(
            benchmark_components["benchmark_portfolio_value"]
        )
        self.state.benchmark_running_peak_value = float(
            benchmark_components["benchmark_running_peak_value"]
        )
        self.state.benchmark_current_drawdown = float(
            benchmark_components["benchmark_current_drawdown"]
        )
        self.state.benchmark_max_drawdown = float(
            benchmark_components["benchmark_max_drawdown"]
        )
        self.state.benchmark_has_rebalanced = True
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
            "constraint_cost": constraint_cost,
            "constraint_mode": self.config.constraint_mode,
            **drawdown_components,
            **benchmark_components,
            **action_components,
            **branch_components,
            "weights": weights.copy(),
            "regime": int(self.market.regimes[self.state.current_index - 1]),
            **constraint_components,
        }
        return observation, reward, terminated, truncated, info
