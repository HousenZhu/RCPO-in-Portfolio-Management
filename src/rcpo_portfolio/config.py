from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    output_root: str = "runs"
    run_name: str = "latest"
    seeds: list[int] = field(default_factory=lambda: [0])


@dataclass
class MarketConfig:
    num_risky_assets: int = 5
    lookback: int = 20
    train_steps: int = 10080
    validation_steps: int = 252
    test_steps: int = 252
    low_vol_drift: float = 0.0007
    high_vol_drift: float = -0.0004
    low_vol_scale: float = 0.008
    high_vol_scale: float = 0.02
    base_correlation: float = 0.25
    transition_matrix: list[list[float]] = field(
        default_factory=lambda: [[0.95, 0.05], [0.10, 0.90]]
    )


@dataclass
class EnvironmentConfig:
    episode_length: int = 252
    transaction_cost_bps: float = 10.0
    turnover_cap: float = 0.40
    constraint_mode: str = "downside"
    downside_cost_scale: float = 1e-4
    sortino_target: float = 1.0
    sortino_window: int = 252
    sortino_min_periods: int = 20
    sortino_cost_scale: float = 1.0
    group_a_indices: list[int] = field(default_factory=lambda: [0, 1])
    group_b_indices: list[int] = field(default_factory=lambda: [2, 3, 4])
    active_constraint_preset: str = "c2"
    constraint_presets: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "c1": {"group_a_min_weight": 0.20, "group_b_max_weight": 0.70},
            "c2": {"group_a_min_weight": 0.25, "group_b_max_weight": 0.60},
            "c3": {"group_a_min_weight": 0.30, "group_b_max_weight": 0.50},
        }
    )
    resolved_group_a_min_weight: float | None = None
    resolved_group_b_max_weight: float | None = None
    downside_cost_weight: float = 1.0
    group_a_min_cost_weight: float = 0.35
    group_b_max_cost_weight: float = 0.35


@dataclass
class NetworkConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    init_log_std: float = -0.5
    min_log_std: float = -1.5


@dataclass
class OptimizationConfig:
    total_updates: int = 200
    rollout_steps: int = 1024
    epochs: int = 8
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 1e-3
    reward_value_coef: float = 0.5
    cost_value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate_final: float | None = None
    target_kl: float | None = None
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0


@dataclass
class PPOOptimizationConfig:
    total_updates: int = 2000
    rollout_steps: int = 2048
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 1e-4
    learning_rate_final: float | None = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    entropy_coef: float = 2e-3
    reward_value_coef: float = 0.5
    cost_value_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    early_stop_patience: int | None = 250
    early_stop_min_delta: float = 0.001


@dataclass
class RCPOConfig:
    initial_lambda: float = 0.0
    lambda_lr: float = 0.05
    alpha: float | None = None
    calibration_episodes: int = 8
    calibration_scale: float = 0.9
    constraint_mode: str = "downside"
    sortino_target: float = 1.0
    sortino_window: int = 252
    sortino_min_periods: int = 20
    sortino_cost_scale: float = 1.0


@dataclass
class EvaluationConfig:
    episodes: int = 5
    deterministic: bool = True
    rolling_risk_window: int = 20


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    ppo: PPOOptimizationConfig = field(default_factory=PPOOptimizationConfig)
    rcpo: RCPOConfig = field(default_factory=RCPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sync_rcpo_constraint_settings(config: ProjectConfig) -> None:
    if config.rcpo.constraint_mode not in {"downside", "sortino"}:
        raise ValueError("rcpo.constraint_mode must be either 'downside' or 'sortino'.")
    config.environment.constraint_mode = config.rcpo.constraint_mode
    config.environment.sortino_target = config.rcpo.sortino_target
    config.environment.sortino_window = config.rcpo.sortino_window
    config.environment.sortino_min_periods = config.rcpo.sortino_min_periods
    config.environment.sortino_cost_scale = config.rcpo.sortino_cost_scale


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    allowed_keys = {field_info.name for field_info in fields(cls)}
    filtered_payload = {key: value for key, value in payload.items() if key in allowed_keys}
    return cls(**filtered_payload)


def _from_dict(payload: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        experiment=_dataclass_from_dict(ExperimentConfig, payload.get("experiment", {})),
        market=_dataclass_from_dict(MarketConfig, payload.get("market", {})),
        environment=_dataclass_from_dict(EnvironmentConfig, payload.get("environment", {})),
        network=_dataclass_from_dict(NetworkConfig, payload.get("network", {})),
        optimization=_dataclass_from_dict(OptimizationConfig, payload.get("optimization", {})),
        ppo=_dataclass_from_dict(PPOOptimizationConfig, payload.get("ppo", {})),
        rcpo=_dataclass_from_dict(RCPOConfig, payload.get("rcpo", {})),
        evaluation=_dataclass_from_dict(EvaluationConfig, payload.get("evaluation", {})),
    )


def load_config(path: str | Path | None = None) -> ProjectConfig:
    defaults = ProjectConfig().to_dict()
    if path is None:
        return _from_dict(defaults)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    merged = _merge_dicts(defaults, payload)
    return _from_dict(merged)


def save_config(config: ProjectConfig, path: str | Path) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
