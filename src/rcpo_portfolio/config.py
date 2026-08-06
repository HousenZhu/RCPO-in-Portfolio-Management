from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION = "benchmark_relative_equal_weight_drawdown_v1"
RELATIVE_CURRENT_DRAWDOWN_CONSTRAINT_VERSION = "relative_current_drawdown_v1"
ALLOCATION_CONSTRAINT_VERSION = "soft_allocation_penalty_v1"
ALLOCATION_DRAWDOWN_CONSTRAINT_VERSION = "soft_allocation_plus_drawdown_penalty_v1"
ALLOCATION_RELATIVE_DRAWDOWN_CONSTRAINT_VERSION = (
    "soft_allocation_plus_relative_current_drawdown_v1"
)


@dataclass
class ExperimentConfig:
    output_root: str = "runs"
    run_name: str = "latest"
    seeds: list[int] = field(default_factory=lambda: [0])


@dataclass
class RuntimeConfig:
    device: str = "auto"


@dataclass
class MarketConfig:
    num_risky_assets: int = 5
    lookback: int = 20
    train_market_count: int = 8
    train_steps: int = 5040
    validation_steps: int = 252
    test_steps: int = 252
    low_vol_drift: float = 0.0007
    high_vol_drift: float = -0.0004
    low_vol_scale: float = 0.008
    high_vol_scale: float = 0.02
    base_correlation: float = 0.25
    low_vol_correlation: float = 0.15
    high_vol_correlation: float = 0.55
    enable_learnable_structure: bool = True
    regime_drift_adjustments: dict[str, list[float]] = field(
        default_factory=lambda: {
            "low_vol": [0.00080, 0.00065, -0.00040, -0.00045, -0.00045],
            "high_vol": [-0.00100, -0.00080, -0.00060, 0.00040, 0.00060],
        }
    )
    momentum_strength: float = 0.20
    momentum_decay: float = 0.94
    momentum_clip: float = 0.01
    transition_matrix: list[list[float]] = field(
        default_factory=lambda: [[0.95, 0.05], [0.10, 0.90]]
    )


@dataclass
class EnvironmentConfig:
    action_mode: str = "softmax"
    simplex_action_format: str = "branch_logits"
    initial_portfolio_mode: str = "all_cash"
    episode_length: int = 252
    transaction_cost_bps: float = 1.0
    turnover_cap: float = 0.40
    constraint_mode: str = "max_drawdown"
    observation_schema_version: int = 1
    drawdown_budget_floor: float = 0.02
    drawdown_benchmark_mode: str = "true_equal_weight"
    benchmark_drawdown_margin: float = 0.90
    drawdown_cost_scale: float = 0.01
    allocation_constraint_cost_scale: float = 20.0
    combined_drawdown_cost_weight: float = 0.25
    diversification_beta: float = 0.03
    allocation_constraint_1_indices: list[int] = field(default_factory=lambda: [1, 2, 4])
    allocation_constraint_2_indices: list[int] = field(default_factory=lambda: [0, 4, 5])
    active_constraint_preset: str = "c2"
    constraint_presets: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "c1": {
                "allocation_constraint_1_min_weight": 0.35,
                "allocation_constraint_2_min_weight": 0.35,
            },
            "c2": {
                "allocation_constraint_1_min_weight": 0.40,
                "allocation_constraint_2_min_weight": 0.40,
            },
            "c3": {
                "allocation_constraint_1_min_weight": 0.55,
                "allocation_constraint_2_min_weight": 0.55,
            },
        }
    )
    resolved_allocation_constraint_1_min_weight: float | None = None
    resolved_allocation_constraint_2_min_weight: float | None = None


@dataclass
class NetworkConfig:
    policy_architecture: str = "flat_gaussian"
    branch_credit_mode: str = "global"
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    init_log_std: float = -1.5
    min_log_std: float = -2.5
    equal_weight_policy_init: bool = True
    dirichlet_min_concentration: float = 0.5
    dirichlet_init_concentration: float = 12.0
    dirichlet_max_concentration: float = 80.0


@dataclass
class OptimizationConfig:
    total_updates: int = 800
    rollout_steps: int = 2048
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 1.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    entropy_coef: float = 5e-4
    reward_value_coef: float = 0.5
    cost_value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate_final: float | None = 3e-5
    target_kl: float | None = 0.02
    early_stop_patience: int | None = 50
    early_stop_min_delta: float = 0.001


@dataclass
class PPOOptimizationConfig:
    total_updates: int = 800
    rollout_steps: int = 2048
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    learning_rate_final: float | None = 2e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    entropy_coef: float = 2e-4
    reward_value_coef: float = 0.5
    cost_value_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    early_stop_patience: int | None = 50
    early_stop_min_delta: float = 0.001


@dataclass
class RCPOConfig:
    initial_lambda: float = 0.0
    lambda_lr: float = 0.02
    lambda_lr_up: float = 0.015
    lambda_lr_down: float = 0.03
    alpha: float | None = None
    alpha_budget_ratio: float = 0.05
    constraint_mode: str = "max_drawdown"


@dataclass
class RewardCorrectionConfig:
    mode: str = "none"
    reward_min: float = -0.04
    reward_max: float = 0.07
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 1e-3
    train_epochs_per_update: int = 3
    num_bins: int = 48
    gdrc_num_candidates: int = 6
    gdrc_candidate_bins: list[int] = field(default_factory=lambda: [48, 64])
    gdrc_vote_decay: float = 0.9
    gdrc_range_window_updates: int = 10
    gdrc_range_percentiles: list[float] = field(default_factory=lambda: [0.5, 99.5])
    correction_coef: float = 0.50
    correction_delta_clip: float = 0.0015


@dataclass
class RewardNoiseConfig:
    enabled: bool = False
    mode: str = "gaussian"
    std: float = 0.003
    seed_offset: int = 30_000


@dataclass
class EvaluationConfig:
    episodes: int = 1
    deterministic: bool = True
    rolling_risk_window: int = 20
    validation_branch_count: int = 5
    test_branch_count: int = 10
    validation_interval_updates: int = 5
    checkpoint_score: str = "validation_mean_excess_cumulative_return"


@dataclass
class LoggingConfig:
    metrics_schema_version: int = 1
    print_interval_updates: int = 1
    branch_diagnostic_interval_updates: int = 50

@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    ppo: PPOOptimizationConfig = field(default_factory=PPOOptimizationConfig)
    rcpo: RCPOConfig = field(default_factory=RCPOConfig)
    reward_correction: RewardCorrectionConfig = field(default_factory=RewardCorrectionConfig)
    reward_noise: RewardNoiseConfig = field(default_factory=RewardNoiseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sync_rcpo_constraint_settings(config: ProjectConfig) -> None:
    valid_policy_architectures = {
        "flat_gaussian",
        "simplex_branch_gaussian",
        "simplex_autoregressive_gaussian",
        "simplex_autoregressive_dirichlet",
    }
    if config.network.policy_architecture not in valid_policy_architectures:
        raise ValueError(
            "network.policy_architecture must be one of: "
            f"{sorted(valid_policy_architectures)}."
        )
    if config.environment.action_mode not in {"softmax", "simplex_decomposition"}:
        raise ValueError(
            "environment.action_mode must be either 'softmax' or 'simplex_decomposition'."
        )
    if (
        config.environment.action_mode != "simplex_decomposition"
        and config.network.policy_architecture != "flat_gaussian"
    ):
        raise ValueError(
            "Simplex branch policy architectures require "
            "environment.action_mode='simplex_decomposition'."
        )
    valid_branch_credit_modes = {
        "global",
        "standalone",
        "standalone_reward_global_cost",
    }
    if config.network.branch_credit_mode not in valid_branch_credit_modes:
        raise ValueError(
            "network.branch_credit_mode must be one of: "
            f"{sorted(valid_branch_credit_modes)}."
        )
    if (
        config.network.branch_credit_mode != "global"
        and config.environment.action_mode != "simplex_decomposition"
    ):
        raise ValueError("Standalone branch credit requires simplex_decomposition action mode.")
    config.environment.simplex_action_format = (
        "branch_weights"
        if config.network.policy_architecture == "simplex_autoregressive_dirichlet"
        else "branch_logits"
    )
    if config.environment.initial_portfolio_mode not in {"all_cash", "constrained_neutral"}:
        raise ValueError(
            "environment.initial_portfolio_mode must be 'all_cash' or 'constrained_neutral'."
        )
    if config.network.dirichlet_min_concentration <= 0.0:
        raise ValueError("network.dirichlet_min_concentration must be positive.")
    if config.network.dirichlet_max_concentration <= config.network.dirichlet_min_concentration:
        raise ValueError(
            "network.dirichlet_max_concentration must exceed dirichlet_min_concentration."
        )
    if not (
        config.network.dirichlet_min_concentration
        <= config.network.dirichlet_init_concentration
        <= config.network.dirichlet_max_concentration
    ):
        raise ValueError(
            "network.dirichlet_init_concentration must lie within the configured bounds."
        )
    valid_constraint_modes = {
        "max_drawdown",
        "relative_current_drawdown",
        "allocation",
        "allocation_drawdown",
        "allocation_relative_drawdown",
    }
    if config.rcpo.constraint_mode not in valid_constraint_modes:
        raise ValueError(
            "rcpo.constraint_mode must be one of: "
            f"{sorted(valid_constraint_modes)}."
        )
    config.environment.constraint_mode = config.rcpo.constraint_mode
    if config.environment.drawdown_budget_floor < 0.0:
        raise ValueError("environment.drawdown_budget_floor cannot be negative.")
    if config.environment.drawdown_benchmark_mode not in {
        "true_equal_weight",
        "constrained_neutral",
    }:
        raise ValueError(
            "environment.drawdown_benchmark_mode must be either "
            "'true_equal_weight' or 'constrained_neutral'."
        )
    if config.environment.benchmark_drawdown_margin <= 0.0:
        raise ValueError("environment.benchmark_drawdown_margin must be positive.")
    if config.environment.drawdown_cost_scale <= 0.0:
        raise ValueError("environment.drawdown_cost_scale must be positive.")
    if config.environment.allocation_constraint_cost_scale <= 0.0:
        raise ValueError("environment.allocation_constraint_cost_scale must be positive.")
    if config.environment.combined_drawdown_cost_weight < 0.0:
        raise ValueError("environment.combined_drawdown_cost_weight cannot be negative.")
    if (
        config.rcpo.constraint_mode in {"allocation", "allocation_drawdown"}
        and config.rcpo.alpha is None
    ):
        raise ValueError(
            "rcpo.alpha must be set when rcpo.constraint_mode is 'allocation' "
            "or 'allocation_drawdown'."
        )
    if config.rcpo.alpha_budget_ratio < 0.0:
        raise ValueError("rcpo.alpha_budget_ratio cannot be negative.")
    if config.rcpo.lambda_lr_up <= 0.0:
        raise ValueError("rcpo.lambda_lr_up must be positive.")
    if config.rcpo.lambda_lr_down <= 0.0:
        raise ValueError("rcpo.lambda_lr_down must be positive.")
    if config.environment.observation_schema_version not in {1, 2}:
        raise ValueError("environment.observation_schema_version must be 1 or 2.")
    if (
        config.rcpo.constraint_mode
        in {"relative_current_drawdown", "allocation_relative_drawdown"}
        and config.environment.observation_schema_version != 2
    ):
        raise ValueError(
            "Relative-current-drawdown modes require observation_schema_version=2."
        )
    if config.logging.metrics_schema_version not in {1, 2}:
        raise ValueError("logging.metrics_schema_version must be 1 or 2.")
    if config.logging.print_interval_updates < 1:
        raise ValueError("logging.print_interval_updates must be positive.")
    if config.logging.branch_diagnostic_interval_updates < 1:
        raise ValueError("logging.branch_diagnostic_interval_updates must be positive.")


def validate_reward_correction_settings(config: ProjectConfig) -> None:
    reward_config = config.reward_correction
    if reward_config.mode not in {"none", "drc", "gdrc"}:
        raise ValueError("reward_correction.mode must be one of: none, drc, gdrc.")
    if reward_config.reward_max <= reward_config.reward_min:
        raise ValueError("reward_correction.reward_max must be greater than reward_min.")
    if reward_config.learning_rate <= 0.0:
        raise ValueError("reward_correction.learning_rate must be positive.")
    if reward_config.train_epochs_per_update < 0:
        raise ValueError("reward_correction.train_epochs_per_update cannot be negative.")
    if reward_config.num_bins < 2:
        raise ValueError("reward_correction.num_bins must be at least 2.")
    if reward_config.gdrc_num_candidates < 1:
        raise ValueError("reward_correction.gdrc_num_candidates must be positive.")
    if not reward_config.gdrc_candidate_bins:
        raise ValueError("reward_correction.gdrc_candidate_bins cannot be empty.")
    if any(int(candidate) < 2 for candidate in reward_config.gdrc_candidate_bins):
        raise ValueError(
            "reward_correction.gdrc_candidate_bins must contain integers at least 2."
        )
    if not 0.0 <= reward_config.gdrc_vote_decay <= 1.0:
        raise ValueError("reward_correction.gdrc_vote_decay must be between 0 and 1.")
    if reward_config.gdrc_range_window_updates < 1:
        raise ValueError("reward_correction.gdrc_range_window_updates must be positive.")
    if len(reward_config.gdrc_range_percentiles) != 2:
        raise ValueError("reward_correction.gdrc_range_percentiles must contain two values.")
    low, high = reward_config.gdrc_range_percentiles
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(
            "reward_correction.gdrc_range_percentiles must satisfy 0 <= low < high <= 100."
        )
    if reward_config.correction_coef < 0.0:
        raise ValueError("reward_correction.correction_coef cannot be negative.")
    if reward_config.correction_delta_clip < 0.0:
        raise ValueError("reward_correction.correction_delta_clip cannot be negative.")
    if config.network.branch_credit_mode != "global" and reward_config.mode != "none":
        raise ValueError(
            "Standalone branch credit currently requires reward_correction.mode='none'."
        )


def validate_reward_noise_settings(config: ProjectConfig) -> None:
    noise_config = config.reward_noise
    if noise_config.mode != "gaussian":
        raise ValueError("reward_noise.mode must be 'gaussian'.")
    if noise_config.std < 0.0:
        raise ValueError("reward_noise.std cannot be negative.")
    if config.network.branch_credit_mode != "global" and noise_config.enabled:
        raise ValueError("Standalone branch credit currently requires clean rewards.")


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
        runtime=_dataclass_from_dict(RuntimeConfig, payload.get("runtime", {})),
        market=_dataclass_from_dict(MarketConfig, payload.get("market", {})),
        environment=_dataclass_from_dict(EnvironmentConfig, payload.get("environment", {})),
        network=_dataclass_from_dict(NetworkConfig, payload.get("network", {})),
        optimization=_dataclass_from_dict(OptimizationConfig, payload.get("optimization", {})),
        ppo=_dataclass_from_dict(PPOOptimizationConfig, payload.get("ppo", {})),
        rcpo=_dataclass_from_dict(RCPOConfig, payload.get("rcpo", {})),
        reward_correction=_dataclass_from_dict(
            RewardCorrectionConfig, payload.get("reward_correction", {})
        ),
        reward_noise=_dataclass_from_dict(
            RewardNoiseConfig, payload.get("reward_noise", {})
        ),
        evaluation=_dataclass_from_dict(EvaluationConfig, payload.get("evaluation", {})),
        logging=_dataclass_from_dict(LoggingConfig, payload.get("logging", {})),
    )


def _load_config_payload(
    config_path: Path,
    ancestry: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved_path = config_path.resolve()
    if resolved_path in ancestry:
        chain = " -> ".join(str(path) for path in (*ancestry, resolved_path))
        raise ValueError(f"Config inheritance cycle detected: {chain}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    base_reference = payload.pop("extends", None)
    if base_reference is None:
        return payload
    if not isinstance(base_reference, str) or not base_reference.strip():
        raise ValueError("Config 'extends' must be a non-empty path string.")

    base_path = Path(base_reference)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_payload = _load_config_payload(
        base_path,
        ancestry=(*ancestry, resolved_path),
    )
    return _merge_dicts(base_payload, payload)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    defaults = ProjectConfig().to_dict()
    if path is None:
        return _from_dict(defaults)
    config_path = Path(path)
    payload = _load_config_payload(config_path)
    merged = _merge_dicts(defaults, payload)
    return _from_dict(merged)


def save_config(config: ProjectConfig, path: str | Path) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
