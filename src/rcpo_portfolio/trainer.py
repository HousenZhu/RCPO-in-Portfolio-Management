from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algorithms import (
    combine_advantages,
    update_lagrange_multiplier,
    update_ppo_actor_critic,
    update_rcpo_actor_critic,
)
from .config import (
    ALLOCATION_CONSTRAINT_VERSION,
    BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION,
    ProjectConfig,
    save_config,
    sync_rcpo_constraint_settings,
    validate_reward_correction_settings,
    validate_reward_noise_settings,
)
from .devices import move_optimizer_state_to_device, resolve_device
from .env import PortfolioEnv
from .env_pool import PortfolioEnvPool
from .evaluation import (
    EvaluationResult,
    evaluate_policy,
    save_evaluation_artifacts,
    save_group_weights_artifact,
    save_training_progress_artifacts,
)
from .io_utils import exclusive_run_lock
from .io_utils import safe_torch_save
from .market import generate_continuation_splits, generate_train_markets
from .models import ActorCritic
from .profiling import TrainingProfiler, profile_section
from .reward_correction import build_reward_corrector
from .rollouts import RolloutBatch, collect_rollout


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RCPOTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        algo: str,
        run_dir: Path,
        seed: int,
        resume_checkpoint: Path | None = None,
        profiler: TrainingProfiler | None = None,
        disable_artifacts: bool = False,
        skip_validation: bool = False,
    ) -> None:
        self.config = config
        self.algo = algo
        self.run_dir = run_dir
        self.seed = seed
        self.resume_checkpoint = resume_checkpoint
        self.profiler = profiler
        self.disable_artifacts = bool(disable_artifacts)
        self.skip_validation = bool(skip_validation)
        self.resume_completed_updates = 0
        set_global_seeds(seed)
        sync_rcpo_constraint_settings(config)
        validate_reward_correction_settings(config)
        validate_reward_noise_settings(config)
        self.device = resolve_device(config.runtime.device)
        self.reward_noise_rng = np.random.default_rng(
            seed + int(config.reward_noise.seed_offset)
        )

        self.train_markets = generate_train_markets(config.market, seed)
        self.train_market = self.train_markets[0]
        self.train_envs = [
            PortfolioEnv(config.environment, market, config.market, seed=seed + 101 * index)
            for index, market in enumerate(self.train_markets)
        ]
        self.train_env = PortfolioEnvPool(self.train_envs, seed=seed)
        self.train_anchor_env = self.train_envs[0]
        self.validation_markets = generate_continuation_splits(
            config.market,
            self.train_market,
            config.market.validation_steps,
            seed + 10_000,
            count=max(1, int(config.evaluation.validation_branch_count)),
        )
        self.test_markets = generate_continuation_splits(
            config.market,
            self.train_market,
            config.market.test_steps,
            seed + 20_000,
            count=max(1, int(config.evaluation.test_branch_count)),
        )
        self.validation_env = PortfolioEnv(
            config.environment, self.validation_markets[0], config.market, seed=seed + 1
        )
        self.test_env = PortfolioEnv(
            config.environment, self.test_markets[0], config.market, seed=seed + 2
        )
        self.optimization = config.ppo if algo == "ppo_unconstrained" else config.optimization

        obs_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.shape[0]
        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.network,
            branch_sizes=self.train_env.simplex_branch_sizes(),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimization.learning_rate)
        self.reward_corrector = build_reward_corrector(
            config.reward_correction,
            obs_dim=obs_dim,
            action_dim=action_dim,
            rollout_steps=self.optimization.rollout_steps,
        ).to(self.device)
        self.lambda_value = 0.0 if algo == "ppo_unconstrained" else config.rcpo.initial_lambda
        self.alpha = self._resolve_alpha()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.lambda_history: list[float] = [float(self.lambda_value)]
        self._resolved_preset = self.train_env.resolved_constraint_preset()
        if self.resume_checkpoint is not None:
            self._load_resume_checkpoint()

    def _print_run_header(self) -> None:
        print(
            f"[train] algo={self.algo} reward_correction={self.config.reward_correction.mode} "
            f"seed={self.seed} updates={self.optimization.total_updates} "
            f"device={self.device} "
            f"train_market_count={len(self.train_markets)} "
            f"validation_branches={len(self.validation_markets)} "
            f"test_branches={len(self.test_markets)} "
            f"rollout_steps={self.optimization.rollout_steps} "
            f"alpha={'dynamic' if self.alpha is None else self.alpha} "
            f"alpha_budget_ratio={self.config.rcpo.alpha_budget_ratio} "
            f"lambda_lr_up={self.config.rcpo.lambda_lr_up} "
            f"lambda_lr_down={self.config.rcpo.lambda_lr_down} "
            f"reward_noise_enabled={self.config.reward_noise.enabled} "
            f"reward_noise_std={self.config.reward_noise.std} "
            f"constraint_mode={self.config.rcpo.constraint_mode} "
            f"action_mode={self.config.environment.action_mode} "
            f"simplex_action_format={self.config.environment.simplex_action_format} "
            f"policy_architecture={self.config.network.policy_architecture} "
            f"branch_credit_mode={self.config.network.branch_credit_mode} "
            f"initial_portfolio_mode={self.config.environment.initial_portfolio_mode} "
            f"branch_sizes={self.train_env.simplex_branch_sizes()} "
            f"drawdown_budget_floor={self.config.environment.drawdown_budget_floor} "
            f"drawdown_benchmark_mode={self.config.environment.drawdown_benchmark_mode} "
            f"benchmark_drawdown_margin={self.config.environment.benchmark_drawdown_margin} "
            f"preset={self._resolved_preset['preset_name']} "
            f"allocation_constraint_1_indices={self.config.environment.allocation_constraint_1_indices} "
            f"allocation_constraint_1_min={self._resolved_preset['allocation_constraint_1_min_weight']} "
            f"allocation_constraint_2_indices={self.config.environment.allocation_constraint_2_indices} "
            f"allocation_constraint_2_min={self._resolved_preset['allocation_constraint_2_min_weight']} "
            f"run_dir={self.run_dir}",
            flush=True,
        )

    def _print_update_summary(
        self,
        update_index: int,
        total_updates: int,
        metric_row: dict[str, Any],
        validation_score: float,
        update_seconds: float,
        elapsed_seconds: float,
        is_best: bool,
    ) -> None:
        best_marker = " best-checkpoint" if is_best else ""
        print(
            (
                f"[train] update={update_index + 1}/{total_updates} "
                f"elapsed={elapsed_seconds:.1f}s update_time={update_seconds:.1f}s "
                f"batch_reward={metric_row['batch_reward_mean']:.6f} "
                f"observed_reward={metric_row['observed_reward_mean']:.6f} "
                f"reward_delta_abs={metric_row['reward_correction_delta_abs_mean']:.6f} "
                f"reward_oce={metric_row['reward_correction_oce']:.6f} "
                f"gdrc_bins={metric_row['gdrc_selected_bins']} "
                f"batch_constraint={metric_row['batch_constraint_cost_mean']:.6f} "
                f"alloc_cost={metric_row['batch_allocation_constraint_cost_mean']:.6f} "
                f"alloc_raw={metric_row['batch_allocation_constraint_raw_cost_mean']:.6f} "
                f"batch_max_drawdown={metric_row['batch_max_drawdown_mean']:.6f} "
                f"batch_benchmark_max_drawdown={metric_row['batch_benchmark_max_drawdown_mean']:.6f} "
                f"batch_budget={metric_row['batch_effective_drawdown_budget_mean']:.6f} "
                f"batch_dd_violation={metric_row['batch_drawdown_violation_mean']:.6f} "
                f"episode_return={metric_row['episode_return_mean']:.6f} "
                f"turnover={metric_row['batch_turnover_mean']:.6f} "
                f"alloc1={metric_row['batch_allocation_constraint_1_weight_mean']:.4f} "
                f"alloc2={metric_row['batch_allocation_constraint_2_weight_mean']:.4f} "
                f"alloc_violation=({metric_row['batch_allocation_constraint_1_violation_cost_mean']:.2e},"
                f"{metric_row['batch_allocation_constraint_2_violation_cost_mean']:.2e}) "
                f"z=({metric_row['batch_simplex_z1_mean']:.3f},"
                f"{metric_row['batch_simplex_z2_mean']:.3f},"
                f"{metric_row['batch_simplex_z3_mean']:.3f},"
                f"{metric_row['batch_simplex_z4_mean']:.3f}) "
                f"policy_loss={metric_row['policy_loss']:.6f} "
                f"value_loss_r={metric_row['reward_value_loss']:.6f} "
                f"value_loss_c={metric_row['cost_value_loss']:.6f} "
                f"lambda={metric_row['lambda_value']:.6f} "
                f"lambda_adv_ratio={metric_row['batch_lambda_cost_advantage_ratio']:.3f} "
                f"lambda_cost_reward={metric_row['lambda_cost_to_reward_ratio']:.3f} "
                f"alpha={metric_row['alpha']:.6f} "
                f"kl={metric_row['approx_kl']:.6f} "
                f"optimizer_steps={metric_row['optimizer_steps_completed']} "
                f"trigger_kl={metric_row['trigger_minibatch_kl']} "
                f"branch_kl=({metric_row['approx_kl_branch_1']:.5f},"
                f"{metric_row['approx_kl_branch_2']:.5f},"
                f"{metric_row['approx_kl_branch_3']:.5f},"
                f"{metric_row['approx_kl_branch_4']:.5f}) "
                f"clip_frac={metric_row['clip_fraction']:.4f} "
                f"lr={metric_row['learning_rate']:.8f} "
                f"val_eval={metric_row['validation_evaluated']} "
                f"val_return={metric_row['validation_annualized_return']:.6f} "
                f"val_excess={metric_row['validation_mean_excess_cumulative_return']:.6f} "
                f"val_win={metric_row['validation_win_rate_vs_equal_weight']:.3f} "
                f"val_max_drawdown={metric_row['validation_max_drawdown']:.6f} "
                f"val_benchmark_max_drawdown={metric_row['validation_benchmark_max_drawdown']:.6f} "
                f"val_budget={metric_row['validation_effective_drawdown_budget']:.6f} "
                f"val_alpha={metric_row['validation_alpha_target']:.6f} "
                f"val_constraint={metric_row['validation_constraint_cost']:.6f} "
                f"val_alloc=({metric_row['validation_allocation_constraint_1_weight']:.4f},"
                f"{metric_row['validation_allocation_constraint_2_weight']:.4f}) "
                f"val_z=({metric_row['validation_simplex_z1']:.3f},"
                f"{metric_row['validation_simplex_z2']:.3f},"
                f"{metric_row['validation_simplex_z3']:.3f},"
                f"{metric_row['validation_simplex_z4']:.3f}) "
                f"score={validation_score:.6f}{best_marker}"
            ),
            flush=True,
        )

    def _equal_weight_action(self) -> np.ndarray:
        return self.train_env.neutral_action()

    def _policy_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = self.model.get_action_and_value(
                observation_tensor, deterministic=deterministic
            )
        return action.squeeze(0).cpu().numpy()

    def _resolve_alpha(self) -> float | None:
        if self.algo == "ppo_unconstrained":
            return 0.0
        if self.config.rcpo.alpha is not None:
            return float(self.config.rcpo.alpha)
        return None

    def _collect_rollout(self) -> RolloutBatch:
        return collect_rollout(
            env=self.train_env,
            model=self.model,
            optimization=self.optimization,
            reward_corrector=self.reward_corrector,
            device=self.device,
            alpha_budget_ratio=(
                self.config.rcpo.alpha_budget_ratio
                if self.algo == "rcpo" and self.alpha is None
                else None
            ),
            drawdown_cost_scale=(
                self.config.environment.drawdown_cost_scale
                if self.algo == "rcpo" and self.alpha is None
                else None
            ),
            reward_noise_config=self.config.reward_noise,
            reward_noise_rng=self.reward_noise_rng,
            profiler=self.profiler,
        )

    def _update_model(self, batch: RolloutBatch) -> dict[str, float]:
        if self.algo == "ppo_unconstrained":
            return update_ppo_actor_critic(
                model=self.model,
                optimizer=self.optimizer,
                batch=batch,
                optimization=self.optimization,
                profiler=self.profiler,
            )
        effective_alpha = (
            self.alpha
            if self.alpha is not None
            else float(batch.info_summary["batch_alpha_target_mean"])
        )
        lambda_gap = float(batch.info_summary["batch_constraint_cost_mean"]) - float(
            effective_alpha
        )
        losses, self.lambda_value, lambda_updates = update_rcpo_actor_critic(
            model=self.model,
            optimizer=self.optimizer,
            batch=batch,
            optimization=self.optimization,
            lambda_value=self.lambda_value,
            alpha=effective_alpha,
            lambda_lr=self.config.rcpo.lambda_lr,
            lambda_lr_up=self.config.rcpo.lambda_lr_up,
            lambda_lr_down=self.config.rcpo.lambda_lr_down,
            profiler=self.profiler,
        )
        losses["lambda_gap"] = lambda_gap
        losses["lambda_lr_up"] = float(self.config.rcpo.lambda_lr_up)
        losses["lambda_lr_down"] = float(self.config.rcpo.lambda_lr_down)
        self.lambda_history.extend(lambda_updates)
        return losses

    def _write_metric_row(self, payload: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _checkpoint(self, name: str) -> None:
        safe_torch_save(
            {
                "algo": self.algo,
                "seed": self.seed,
                "alpha": self.alpha,
                "alpha_budget_ratio": float(self.config.rcpo.alpha_budget_ratio),
                "constraint_mode": self.config.rcpo.constraint_mode,
                "constraint_semantics": (
                    ALLOCATION_CONSTRAINT_VERSION
                    if self.config.rcpo.constraint_mode == "allocation"
                    else BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION
                ),
                "drawdown_budget_floor": float(self.config.environment.drawdown_budget_floor),
                "drawdown_benchmark_mode": self.config.environment.drawdown_benchmark_mode,
                "benchmark_drawdown_margin": float(
                    self.config.environment.benchmark_drawdown_margin
                ),
                "drawdown_cost_scale": float(self.config.environment.drawdown_cost_scale),
                "allocation_constraint_cost_scale": float(
                    self.config.environment.allocation_constraint_cost_scale
                ),
                "action_mode": self.config.environment.action_mode,
                "simplex_action_format": self.config.environment.simplex_action_format,
                "action_dim": int(self.train_env.action_space.shape[0]),
                "simplex_branch_sizes": self.train_env.simplex_branch_sizes(),
                "policy_architecture": self.config.network.policy_architecture,
                "branch_credit_mode": self.config.network.branch_credit_mode,
                "initial_portfolio_mode": self.config.environment.initial_portfolio_mode,
                "dirichlet_min_concentration": float(
                    self.config.network.dirichlet_min_concentration
                ),
                "dirichlet_init_concentration": float(
                    self.config.network.dirichlet_init_concentration
                ),
                "dirichlet_max_concentration": float(
                    self.config.network.dirichlet_max_concentration
                ),
                "allocation_constraint_1_indices": list(
                    self.config.environment.allocation_constraint_1_indices
                ),
                "allocation_constraint_2_indices": list(
                    self.config.environment.allocation_constraint_2_indices
                ),
                "active_constraint_preset": self.config.environment.active_constraint_preset,
                "allocation_constraint_1_min_weight": float(
                    self._resolved_preset["allocation_constraint_1_min_weight"]
                ),
                "allocation_constraint_2_min_weight": float(
                    self._resolved_preset["allocation_constraint_2_min_weight"]
                ),
                "lambda_lr_up": float(self.config.rcpo.lambda_lr_up),
                "lambda_lr_down": float(self.config.rcpo.lambda_lr_down),
                "reward_noise_enabled": bool(self.config.reward_noise.enabled),
                "reward_noise_mode": self.config.reward_noise.mode,
                "reward_noise_std": float(self.config.reward_noise.std),
                "reward_noise_seed_offset": int(self.config.reward_noise.seed_offset),
                "reward_noise_rng_state": self.reward_noise_rng.bit_generator.state,
                "reward_correction_mode": self.config.reward_correction.mode,
                "device": str(self.device),
                "lambda_value": self.lambda_value,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "reward_corrector_state_dict": self.reward_corrector.state_dict(),
                "completed_updates": self.resume_completed_updates,
            },
            self.run_dir / name,
        )

    def _load_resume_checkpoint(self) -> None:
        if self.resume_checkpoint is None:
            return
        checkpoint = torch.load(self.resume_checkpoint, map_location=self.device)
        checkpoint_algo = checkpoint.get("algo")
        checkpoint_seed = checkpoint.get("seed")
        if checkpoint_algo is not None and checkpoint_algo != self.algo:
            raise ValueError(
                f"Checkpoint algorithm {checkpoint_algo!r} does not match requested {self.algo!r}."
            )
        if checkpoint_seed is not None and int(checkpoint_seed) != self.seed:
            raise ValueError(
                f"Checkpoint seed {checkpoint_seed!r} does not match requested seed {self.seed!r}."
            )
        checkpoint_action_mode = checkpoint.get("action_mode", "softmax")
        if checkpoint_action_mode != self.config.environment.action_mode:
            raise ValueError(
                f"Checkpoint action_mode {checkpoint_action_mode!r} does not match "
                f"requested {self.config.environment.action_mode!r}."
            )
        checkpoint_simplex_action_format = checkpoint.get(
            "simplex_action_format",
            "branch_logits",
        )
        if checkpoint_simplex_action_format != self.config.environment.simplex_action_format:
            raise ValueError(
                f"Checkpoint simplex_action_format {checkpoint_simplex_action_format!r} "
                f"does not match requested {self.config.environment.simplex_action_format!r}."
            )
        checkpoint_policy_architecture = checkpoint.get("policy_architecture", "flat_gaussian")
        if checkpoint_policy_architecture != self.config.network.policy_architecture:
            raise ValueError(
                f"Checkpoint policy_architecture {checkpoint_policy_architecture!r} "
                f"does not match requested {self.config.network.policy_architecture!r}."
            )
        checkpoint_branch_credit_mode = checkpoint.get("branch_credit_mode", "global")
        if checkpoint_branch_credit_mode != self.config.network.branch_credit_mode:
            raise ValueError(
                f"Checkpoint branch_credit_mode {checkpoint_branch_credit_mode!r} "
                f"does not match requested {self.config.network.branch_credit_mode!r}."
            )
        checkpoint_initial_portfolio_mode = checkpoint.get(
            "initial_portfolio_mode",
            "all_cash",
        )
        if checkpoint_initial_portfolio_mode != self.config.environment.initial_portfolio_mode:
            raise ValueError(
                f"Checkpoint initial_portfolio_mode {checkpoint_initial_portfolio_mode!r} "
                f"does not match requested {self.config.environment.initial_portfolio_mode!r}."
            )
        if self.config.network.policy_architecture == "simplex_autoregressive_dirichlet":
            for field_name in (
                "dirichlet_min_concentration",
                "dirichlet_init_concentration",
                "dirichlet_max_concentration",
            ):
                checkpoint_value = checkpoint.get(field_name)
                configured_value = float(getattr(self.config.network, field_name))
                if checkpoint_value is None or not math.isclose(
                    float(checkpoint_value),
                    configured_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Checkpoint {field_name} does not match the current config."
                    )
        checkpoint_action_dim = checkpoint.get("action_dim")
        if checkpoint_action_dim is not None and int(checkpoint_action_dim) != int(
            self.train_env.action_space.shape[0]
        ):
            raise ValueError(
                f"Checkpoint action_dim {checkpoint_action_dim!r} does not match current "
                f"action dimension {self.train_env.action_space.shape[0]}."
            )
        if (
            self.config.environment.action_mode == "simplex_decomposition"
            or self.config.rcpo.constraint_mode == "allocation"
        ):
            checkpoint_branch_sizes = checkpoint.get("simplex_branch_sizes")
            if (
                self.config.environment.action_mode == "simplex_decomposition"
                and checkpoint_branch_sizes is not None
                and list(checkpoint_branch_sizes) != self.train_env.simplex_branch_sizes()
            ):
                raise ValueError(
                    "Checkpoint simplex_branch_sizes do not match current config."
                )
            expected_c1 = list(self.config.environment.allocation_constraint_1_indices)
            expected_c2 = list(self.config.environment.allocation_constraint_2_indices)
            if checkpoint.get("allocation_constraint_1_indices") != expected_c1:
                raise ValueError(
                    "Checkpoint allocation_constraint_1_indices do not match current config."
                )
            if checkpoint.get("allocation_constraint_2_indices") != expected_c2:
                raise ValueError(
                    "Checkpoint allocation_constraint_2_indices do not match current config."
                )
            if checkpoint.get("active_constraint_preset") != self.config.environment.active_constraint_preset:
                raise ValueError(
                    "Checkpoint active_constraint_preset does not match current config."
                )
            if not math.isclose(
                float(checkpoint.get("allocation_constraint_1_min_weight", math.nan)),
                float(self._resolved_preset["allocation_constraint_1_min_weight"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Checkpoint allocation_constraint_1_min_weight does not match current config."
                )
            if not math.isclose(
                float(checkpoint.get("allocation_constraint_2_min_weight", math.nan)),
                float(self._resolved_preset["allocation_constraint_2_min_weight"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Checkpoint allocation_constraint_2_min_weight does not match current config."
                )
        checkpoint_constraint_mode = checkpoint.get("constraint_mode")
        if self.algo == "rcpo":
            if checkpoint_constraint_mode is None:
                raise ValueError(
                    "RCPO resume requires checkpoints saved with a supported constraint_mode. "
                    "Legacy downside/sortino checkpoints are not supported."
                )
            if checkpoint_constraint_mode != self.config.rcpo.constraint_mode:
                raise ValueError(
                    f"Checkpoint constraint mode {checkpoint_constraint_mode!r} does not match "
                    f"requested {self.config.rcpo.constraint_mode!r}. Legacy downside/sortino "
                    "checkpoints are not supported."
                )
            if self.config.rcpo.constraint_mode == "max_drawdown":
                checkpoint_semantics = checkpoint.get("constraint_semantics")
                if checkpoint_semantics != BENCHMARK_DRAWDOWN_CONSTRAINT_VERSION:
                    raise ValueError(
                        "RCPO resume requires checkpoints saved with the benchmark-relative "
                        "drawdown constraint semantics. Legacy fixed-budget drawdown checkpoints "
                        "are not supported."
                    )
                if not math.isclose(
                    float(checkpoint.get("drawdown_budget_floor", math.nan)),
                    float(self.config.environment.drawdown_budget_floor),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Checkpoint drawdown_budget_floor does not match the current config."
                    )
                checkpoint_benchmark_mode = checkpoint.get(
                    "drawdown_benchmark_mode",
                    "true_equal_weight",
                )
                if checkpoint_benchmark_mode != self.config.environment.drawdown_benchmark_mode:
                    raise ValueError(
                        "Checkpoint drawdown_benchmark_mode does not match the current config."
                    )
                if not math.isclose(
                    float(checkpoint.get("benchmark_drawdown_margin", math.nan)),
                    float(self.config.environment.benchmark_drawdown_margin),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Checkpoint benchmark_drawdown_margin does not match the current config."
                    )
                if not math.isclose(
                    float(checkpoint.get("drawdown_cost_scale", math.nan)),
                    float(self.config.environment.drawdown_cost_scale),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Checkpoint drawdown_cost_scale does not match the current config."
                    )
            if self.config.rcpo.constraint_mode == "allocation":
                checkpoint_semantics = checkpoint.get("constraint_semantics")
                if checkpoint_semantics != ALLOCATION_CONSTRAINT_VERSION:
                    raise ValueError(
                        "RCPO resume requires checkpoints saved with the soft allocation "
                        "constraint semantics."
                    )
                if not math.isclose(
                    float(checkpoint.get("allocation_constraint_cost_scale", math.nan)),
                    float(self.config.environment.allocation_constraint_cost_scale),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Checkpoint allocation_constraint_cost_scale does not match the current config."
                    )
        checkpoint_reward_mode = checkpoint.get("reward_correction_mode", "none")
        if checkpoint_reward_mode != self.config.reward_correction.mode:
            raise ValueError(
                f"Checkpoint reward correction mode {checkpoint_reward_mode!r} does not match "
                f"requested {self.config.reward_correction.mode!r}."
            )
        checkpoint_alpha_budget_ratio = checkpoint.get("alpha_budget_ratio")
        if checkpoint_alpha_budget_ratio is not None and not math.isclose(
            float(checkpoint_alpha_budget_ratio),
            float(self.config.rcpo.alpha_budget_ratio),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Checkpoint alpha_budget_ratio does not match the current config."
            )
        checkpoint_noise_enabled = checkpoint.get("reward_noise_enabled")
        if checkpoint_noise_enabled is not None and bool(checkpoint_noise_enabled) != bool(
            self.config.reward_noise.enabled
        ):
            raise ValueError(
                "Checkpoint reward_noise.enabled does not match the current config."
            )
        checkpoint_noise_std = checkpoint.get("reward_noise_std")
        if checkpoint_noise_std is not None and not math.isclose(
            float(checkpoint_noise_std),
            float(self.config.reward_noise.std),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Checkpoint reward_noise.std does not match the current config.")
        if "reward_noise_rng_state" in checkpoint:
            self.reward_noise_rng.bit_generator.state = checkpoint["reward_noise_rng_state"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(self.optimizer, self.device)
        else:
            print(
                "[train] resume_warning=checkpoint_has_no_optimizer_state using_fresh_optimizer=true",
                flush=True,
            )
        self.reward_corrector.load_state_dict(
            checkpoint.get("reward_corrector_state_dict", {})
        )
        self.reward_corrector.to(self.device)
        if checkpoint.get("alpha") is not None:
            self.alpha = float(checkpoint["alpha"])
        self.lambda_value = float(checkpoint.get("lambda_value", self.lambda_value))
        self.resume_completed_updates = int(checkpoint.get("completed_updates", 0))

    def _read_metric_rows(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if self.resume_checkpoint is not None and rows:
            by_update: dict[int, dict[str, Any]] = {}
            for row in rows:
                update = int(row.get("update", -1))
                if 0 <= update < self.resume_completed_updates:
                    by_update[update] = row
            normalized_rows = [by_update[update] for update in sorted(by_update)]
            original_updates = [int(row.get("update", -1)) for row in rows]
            normalized_updates = [int(row["update"]) for row in normalized_rows]
            if original_updates != normalized_updates:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.metrics_path.with_name(
                    f"metrics_before_resume_repair_{timestamp}.jsonl"
                )
                shutil.copy2(self.metrics_path, backup_path)
                with self.metrics_path.open("w", encoding="utf-8") as handle:
                    for row in normalized_rows:
                        handle.write(json.dumps(row) + "\n")
                print(
                    f"[train] repaired_metrics duplicates_or_ordering=true "
                    f"rows_before={len(rows)} rows_after={len(normalized_rows)} "
                    f"backup={backup_path}",
                    flush=True,
                )
                rows = normalized_rows
        return rows

    def _best_from_metric_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[float, dict[str, Any] | None]:
        best_score = -math.inf
        best_summary: dict[str, Any] | None = None
        for row in rows:
            if "validation_annualized_return" not in row:
                continue
            validation_summary = {
                key[len("validation_") :]: value
                for key, value in row.items()
                if key.startswith("validation_")
            }
            validation_summary["annualized_return"] = row["validation_annualized_return"]
            validation_summary["split"] = "validation"
            score = self._validation_score(validation_summary)
            if score > best_score:
                best_score = score
                best_summary = validation_summary
        return best_score, best_summary

    def _set_learning_rate(self, update_index: int, total_updates: int) -> float:
        start_lr = float(self.optimization.learning_rate)
        final_lr = self.optimization.learning_rate_final
        if self.algo != "ppo_unconstrained" or final_lr is None or total_updates <= 1:
            learning_rate = start_lr
        else:
            progress = update_index / float(total_updates - 1)
            learning_rate = start_lr + progress * (float(final_lr) - start_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def _validation_score(self, validation_summary: dict[str, Any]) -> float:
        score_key = self.config.evaluation.checkpoint_score
        if score_key.startswith("validation_"):
            score_key = score_key[len("validation_") :]
        return float(
            validation_summary.get(
                score_key,
                validation_summary.get("annualized_return", -math.inf),
            )
        )

    def _should_evaluate_validation(
        self,
        local_update_index: int,
        update_index: int,
    ) -> bool:
        if self.skip_validation:
            return False
        interval = max(1, int(self.config.evaluation.validation_interval_updates))
        return local_update_index == 0 or (update_index + 1) % interval == 0

    def _validation_placeholder(self) -> dict[str, Any]:
        return {
            "annualized_return": 0.0,
            "mean_annualized_return": 0.0,
            "mean_cumulative_return": 0.0,
            "equal_weight_mean_cumulative_return": 0.0,
            "mean_excess_cumulative_return": 0.0,
            "win_rate_vs_equal_weight": 0.0,
            "return_std": 0.0,
            "branches": 0,
            "max_drawdown": 0.0,
            "benchmark_max_drawdown": 0.0,
            "drawdown_benchmark_mode": self.config.environment.drawdown_benchmark_mode,
            "effective_drawdown_budget": 0.0,
            "average_alpha_target": 0.0,
            "average_turnover": 0.0,
            "average_constraint_cost": 0.0,
            "average_drawdown_gap": 0.0,
            "average_drawdown_violation": 0.0,
            "average_drawdown_constraint_cost": 0.0,
            "average_allocation_constraint_1_weight": 0.0,
            "average_allocation_constraint_2_weight": 0.0,
            "average_allocation_constraint_1_violation_cost": 0.0,
            "average_allocation_constraint_2_violation_cost": 0.0,
            "average_allocation_constraint_raw_cost": 0.0,
            "average_allocation_constraint_cost": 0.0,
            "average_simplex_z1": 0.0,
            "average_simplex_z2": 0.0,
            "average_simplex_z3": 0.0,
            "average_simplex_z4": 0.0,
            "constraint_violation_rate": 0.0,
            "split": "validation",
        }

    @staticmethod
    def _cumulative_return(returns: np.ndarray) -> float:
        return float(np.prod(1.0 + np.asarray(returns, dtype=np.float64)) - 1.0)

    def _branch_markets(self, split_name: str) -> list:
        if split_name == "validation":
            return self.validation_markets
        if split_name == "test":
            return self.test_markets
        raise ValueError(f"Unsupported branch split: {split_name}")

    def _evaluate_branch_set(
        self,
        split_name: str,
        policy_fn,
    ) -> tuple[EvaluationResult, list[np.ndarray], list[np.ndarray]]:
        branch_results: list[EvaluationResult] = []
        model_returns: list[np.ndarray] = []
        equal_weight_returns: list[np.ndarray] = []
        model_cumulative_returns: list[float] = []
        equal_weight_cumulative_returns: list[float] = []
        seed_offset = 10_000 if split_name == "validation" else 20_000
        for index, market in enumerate(self._branch_markets(split_name)):
            env = PortfolioEnv(
                self.config.environment,
                market,
                self.config.market,
                seed=self.seed + seed_offset + index,
            )
            result = evaluate_policy(
                env,
                policy_fn=policy_fn,
                episodes=1,
                alpha=self.alpha,
                alpha_budget_ratio=(
                    self.config.rcpo.alpha_budget_ratio if self.alpha is None else None
                ),
                split_name=split_name,
            )
            model_path = result.episode_returns[0]
            equal_weight_path = result.equal_weight_episode_returns[0]
            branch_results.append(result)
            model_returns.append(model_path)
            equal_weight_returns.append(equal_weight_path)
            model_cumulative_returns.append(self._cumulative_return(model_path))
            equal_weight_cumulative_returns.append(self._cumulative_return(equal_weight_path))

        if not branch_results:
            raise RuntimeError(f"No {split_name} branches were available for evaluation.")
        summaries = [result.summary for result in branch_results]
        aggregate = {
            metric: float(np.mean([summary[metric] for summary in summaries]))
            for metric in summaries[0]
            if metric != "split" and isinstance(summaries[0][metric], (int, float))
        }
        model_cumulative = np.asarray(model_cumulative_returns, dtype=np.float64)
        equal_weight_cumulative = np.asarray(equal_weight_cumulative_returns, dtype=np.float64)
        aggregate.update(
            {
                "mean_annualized_return": float(aggregate["annualized_return"]),
                "mean_cumulative_return": float(np.mean(model_cumulative)),
                "equal_weight_mean_cumulative_return": float(np.mean(equal_weight_cumulative)),
                "mean_excess_cumulative_return": float(
                    np.mean(model_cumulative - equal_weight_cumulative)
                ),
                "win_rate_vs_equal_weight": float(
                    np.mean(model_cumulative > equal_weight_cumulative)
                ),
                "return_std": float(np.std(model_cumulative)),
                "branches": len(branch_results),
                "drawdown_benchmark_mode": self.config.environment.drawdown_benchmark_mode,
                "split": split_name,
            }
        )
        first_result = branch_results[0]
        aggregate_result = EvaluationResult(
            summary=aggregate,
            first_episode=first_result.first_episode,
            episode_returns=model_returns,
            equal_weight_first_episode_returns=first_result.equal_weight_first_episode_returns,
            equal_weight_episode_returns=equal_weight_returns,
            branch_first_episodes=[result.first_episode for result in branch_results],
        )
        return aggregate_result, model_returns, equal_weight_returns

    @staticmethod
    def _validation_metric_fields(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "validation_annualized_return": summary["annualized_return"],
            "validation_mean_annualized_return": summary["mean_annualized_return"],
            "validation_mean_cumulative_return": summary["mean_cumulative_return"],
            "validation_equal_weight_mean_cumulative_return": summary[
                "equal_weight_mean_cumulative_return"
            ],
            "validation_mean_excess_cumulative_return": summary[
                "mean_excess_cumulative_return"
            ],
            "validation_win_rate_vs_equal_weight": summary["win_rate_vs_equal_weight"],
            "validation_return_std": summary["return_std"],
            "validation_branches": summary["branches"],
            "validation_max_drawdown": summary["max_drawdown"],
            "validation_benchmark_max_drawdown": summary["benchmark_max_drawdown"],
            "validation_drawdown_benchmark_mode": summary["drawdown_benchmark_mode"],
            "validation_effective_drawdown_budget": summary["effective_drawdown_budget"],
            "validation_alpha_target": summary["average_alpha_target"],
            "validation_turnover": summary["average_turnover"],
            "validation_constraint_cost": summary["average_constraint_cost"],
            "validation_drawdown_gap": summary["average_drawdown_gap"],
            "validation_drawdown_violation": summary["average_drawdown_violation"],
            "validation_drawdown_constraint_cost": summary[
                "average_drawdown_constraint_cost"
            ],
            "validation_allocation_constraint_1_weight": summary[
                "average_allocation_constraint_1_weight"
            ],
            "validation_allocation_constraint_2_weight": summary[
                "average_allocation_constraint_2_weight"
            ],
            "validation_allocation_constraint_1_violation_cost": summary[
                "average_allocation_constraint_1_violation_cost"
            ],
            "validation_allocation_constraint_2_violation_cost": summary[
                "average_allocation_constraint_2_violation_cost"
            ],
            "validation_allocation_constraint_raw_cost": summary[
                "average_allocation_constraint_raw_cost"
            ],
            "validation_allocation_constraint_cost": summary[
                "average_allocation_constraint_cost"
            ],
            "validation_simplex_z1": summary["average_simplex_z1"],
            "validation_simplex_z2": summary["average_simplex_z2"],
            "validation_simplex_z3": summary["average_simplex_z3"],
            "validation_simplex_z4": summary["average_simplex_z4"],
            "validation_constraint_violation_rate": summary["constraint_violation_rate"],
        }

    def _evaluate_checkpoint_artifacts(
        self,
        checkpoint_name: str,
        output_dir: Path,
        lambda_history: list[float] | None = None,
        lambda_update_steps: list[int] | None = None,
    ) -> dict[str, Any]:
        checkpoint = torch.load(self.run_dir / checkpoint_name, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        summaries: dict[str, Any] = {}
        for split_name in ["validation", "test"]:
            result, mean_model_returns, mean_equal_weight_returns = self._evaluate_branch_set(
                split_name=split_name,
                policy_fn=lambda obs: self._policy_action(obs, deterministic=True),
            )
            save_evaluation_artifacts(
                result,
                output_dir,
                split_name=split_name,
                rolling_window=self.config.evaluation.rolling_risk_window,
                lambda_history=lambda_history,
                lambda_update_steps=lambda_update_steps,
                mean_episode_returns=mean_model_returns,
                equal_weight_mean_episode_returns=mean_equal_weight_returns,
            )
            summaries[split_name] = result.summary
        return summaries

    @staticmethod
    def _lambda_plot_series(
        metrics_rows: list[dict[str, Any]],
    ) -> tuple[list[float], list[int]]:
        return (
            [float(row.get("lambda_value", 0.0)) for row in metrics_rows],
            [int(row.get("update", index)) + 1 for index, row in enumerate(metrics_rows)],
        )

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config.environment.resolved_allocation_constraint_1_min_weight = float(
            self._resolved_preset["allocation_constraint_1_min_weight"]
        )
        self.config.environment.resolved_allocation_constraint_2_min_weight = float(
            self._resolved_preset["allocation_constraint_2_min_weight"]
        )
        save_config(self.config, self.run_dir / "config_snapshot.yaml")
        self._print_run_header()

        if self.algo == "equal_weight":
            return self._evaluate_equal_weight()

        metrics_rows: list[dict[str, Any]] = (
            self._read_metric_rows() if self.resume_checkpoint is not None else []
        )
        best_score, best_summary = self._best_from_metric_rows(metrics_rows)
        training_start_time = time.perf_counter()
        additional_updates = self.optimization.total_updates
        start_update = max(self.resume_completed_updates, len(metrics_rows))
        target_total_updates = start_update + additional_updates
        if metrics_rows:
            self.lambda_history = [
                float(row.get("lambda_value", self.lambda_value)) for row in metrics_rows
            ]
        last_validation_summary = best_summary
        if self.skip_validation and last_validation_summary is None:
            last_validation_summary = self._validation_placeholder()
            best_summary = last_validation_summary
            best_score = self._validation_score(last_validation_summary)
        if metrics_rows:
            for row in reversed(metrics_rows):
                if "validation_annualized_return" in row:
                    last_validation_summary = {
                        key[len("validation_") :]: value
                        for key, value in row.items()
                        if key.startswith("validation_")
                    }
                    last_validation_summary["annualized_return"] = row[
                        "validation_annualized_return"
                    ]
                    last_validation_summary["split"] = "validation"
                    break
        if self.resume_checkpoint is not None:
            print(
                f"[train] resume_from={self.resume_checkpoint} "
                f"start_update={start_update} additional_updates={additional_updates}",
                flush=True,
            )

        stale_updates = 0
        early_stopped = False
        for local_update_index in range(additional_updates):
            update_index = start_update + local_update_index
            update_start_time = time.perf_counter()
            current_learning_rate = self._set_learning_rate(
                update_index, target_total_updates
            )
            with profile_section(self.profiler, "rollout_total"):
                rollout = self._collect_rollout()
            with profile_section(self.profiler, "optimization_total"):
                losses = self._update_model(rollout)
            validation_evaluated = self._should_evaluate_validation(
                local_update_index,
                update_index,
            )
            if validation_evaluated:
                with profile_section(self.profiler, "validation_total"):
                    validation_result, _, _ = self._evaluate_branch_set(
                        split_name="validation",
                        policy_fn=lambda obs: self._policy_action(obs, deterministic=True),
                    )
                if not self.disable_artifacts:
                    with profile_section(self.profiler, "live_group_weights_plot"):
                        save_group_weights_artifact(
                            validation_result,
                            self.run_dir / "evaluation",
                            "validation",
                            update_number=update_index + 1,
                        )
                last_validation_summary = validation_result.summary
            if last_validation_summary is None:
                if self.skip_validation:
                    last_validation_summary = self._validation_placeholder()
                else:
                    with profile_section(self.profiler, "validation_total"):
                        validation_result, _, _ = self._evaluate_branch_set(
                            split_name="validation",
                            policy_fn=lambda obs: self._policy_action(obs, deterministic=True),
                        )
                    if not self.disable_artifacts:
                        with profile_section(self.profiler, "live_group_weights_plot"):
                            save_group_weights_artifact(
                                validation_result,
                                self.run_dir / "evaluation",
                                "validation",
                                update_number=update_index + 1,
                            )
                    last_validation_summary = validation_result.summary
                    validation_evaluated = True
            validation_score = self._validation_score(last_validation_summary)
            metric_row = {
                "update": update_index,
                "algo": self.algo,
                "alpha": (
                    self.alpha
                    if self.alpha is not None
                    else float(rollout.info_summary["batch_alpha_target_mean"])
                ),
                "alpha_mode": "fixed" if self.alpha is not None else "budget_ratio",
                "alpha_budget_ratio": float(self.config.rcpo.alpha_budget_ratio),
                "constraint_mode": self.config.rcpo.constraint_mode,
                "drawdown_benchmark_mode": self.config.environment.drawdown_benchmark_mode,
                "action_mode": self.config.environment.action_mode,
                "simplex_action_format": self.config.environment.simplex_action_format,
                "policy_architecture": self.config.network.policy_architecture,
                "branch_credit_mode": self.config.network.branch_credit_mode,
                "initial_portfolio_mode": self.config.environment.initial_portfolio_mode,
                "reward_correction_mode": self.config.reward_correction.mode,
                "lambda_value": self.lambda_value,
                "learning_rate": current_learning_rate,
                "constraint_preset": self._resolved_preset["preset_name"],
                "device": str(self.device),
                "turnover_cap": self.config.environment.turnover_cap,
                "validation_evaluated": int(validation_evaluated),
                "validation_interval_updates": int(
                    self.config.evaluation.validation_interval_updates
                ),
                **rollout.info_summary,
                **losses,
                **self._validation_metric_fields(last_validation_summary),
            }
            reward_advantage_std = float(metric_row.get("batch_reward_advantage_std", 0.0))
            cost_advantage_std = float(metric_row.get("batch_cost_advantage_std", 0.0))
            batch_reward_mean = float(metric_row.get("batch_reward_mean", 0.0))
            batch_constraint_mean = float(metric_row.get("batch_constraint_cost_mean", 0.0))
            lambda_value = float(metric_row.get("lambda_value", 0.0))
            metric_row["batch_lambda_cost_advantage_ratio"] = float(
                abs(lambda_value) * cost_advantage_std / max(reward_advantage_std, 1e-12)
            )
            metric_row["lambda_cost_to_reward_ratio"] = float(
                abs(lambda_value) * abs(batch_constraint_mean) / max(abs(batch_reward_mean), 1e-12)
            )
            with profile_section(self.profiler, "metric_write"):
                self._write_metric_row(metric_row)
            metrics_rows.append(metric_row)

            is_best = False
            previous_best_score = best_score
            min_delta = self.optimization.early_stop_min_delta
            if validation_evaluated and validation_score > best_score:
                best_score = validation_score
                best_summary = last_validation_summary
                with profile_section(self.profiler, "checkpoint_best"):
                    self._checkpoint("checkpoint_best.pt")
                is_best = True
            if validation_evaluated and validation_score > previous_best_score + min_delta:
                stale_updates = 0
            elif validation_evaluated:
                stale_updates += 1
            self.resume_completed_updates = update_index + 1
            with profile_section(self.profiler, "checkpoint_last"):
                self._checkpoint("checkpoint_last.pt")
            with profile_section(self.profiler, "print_summary"):
                self._print_update_summary(
                    update_index=update_index,
                    total_updates=target_total_updates,
                    metric_row=metric_row,
                    validation_score=validation_score,
                    update_seconds=time.perf_counter() - update_start_time,
                    elapsed_seconds=time.perf_counter() - training_start_time,
                    is_best=is_best,
                )
            if (
                self.optimization.early_stop_patience is not None
                and stale_updates >= self.optimization.early_stop_patience
            ):
                early_stopped = True
                print(
                    f"[train] early_stop=validation_score "
                    f"patience={self.optimization.early_stop_patience} "
                    f"completed_updates={self.resume_completed_updates}",
                    flush=True,
                )
                break

        if best_summary is None:
            raise RuntimeError("Training did not produce a validation summary.")
        if self.disable_artifacts:
            best_evaluation = {}
            last_evaluation = {}
        else:
            save_training_progress_artifacts(metrics_rows, self.run_dir / "evaluation")
            lambda_plot_history, lambda_plot_steps = self._lambda_plot_series(metrics_rows)
            best_checkpoint = self.run_dir / "checkpoint_best.pt"
            if best_checkpoint.exists():
                best_evaluation = self._evaluate_checkpoint_artifacts(
                    "checkpoint_best.pt",
                    self.run_dir / "evaluation_best",
                    lambda_history=lambda_plot_history,
                    lambda_update_steps=lambda_plot_steps,
                )
            else:
                best_evaluation = {}
            last_evaluation = self._evaluate_checkpoint_artifacts(
                "checkpoint_last.pt",
                self.run_dir / "evaluation_last",
                lambda_history=lambda_plot_history,
                lambda_update_steps=lambda_plot_steps,
            )
        summary = {
            "algo": self.algo,
            "seed": self.seed,
            "alpha": self.alpha,
            "alpha_budget_ratio": float(self.config.rcpo.alpha_budget_ratio),
            "lambda_lr_up": float(self.config.rcpo.lambda_lr_up),
            "lambda_lr_down": float(self.config.rcpo.lambda_lr_down),
            "reward_noise_enabled": bool(self.config.reward_noise.enabled),
            "reward_noise_mode": self.config.reward_noise.mode,
            "reward_noise_std": float(self.config.reward_noise.std),
            "constraint_mode": self.config.rcpo.constraint_mode,
            "drawdown_benchmark_mode": self.config.environment.drawdown_benchmark_mode,
            "action_mode": self.config.environment.action_mode,
            "simplex_action_format": self.config.environment.simplex_action_format,
            "policy_architecture": self.config.network.policy_architecture,
            "branch_credit_mode": self.config.network.branch_credit_mode,
            "initial_portfolio_mode": self.config.environment.initial_portfolio_mode,
            "reward_correction_mode": self.config.reward_correction.mode,
            "device": str(self.device),
            "constraint_preset": self._resolved_preset["preset_name"],
            "best_validation": best_summary,
            "evaluation_best": best_evaluation,
            "evaluation_last": last_evaluation,
            "lambda_final": self.lambda_value,
            "completed_updates": len(metrics_rows),
            "early_stopped": early_stopped,
        }
        with (self.run_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def _evaluate_equal_weight(self) -> dict[str, Any]:
        evaluation_dir = self.run_dir / "evaluation"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, self.run_dir / "config_snapshot.yaml")
        summaries: dict[str, Any] = {}
        for split_name, env in [
            ("train", self.train_env),
            ("validation", self.validation_env),
            ("test", self.test_env),
        ]:
            result = evaluate_policy(
                env,
                policy_fn=lambda _obs: self._equal_weight_action(),
                episodes=self.config.evaluation.episodes,
                alpha=self.alpha,
                alpha_budget_ratio=(
                    self.config.rcpo.alpha_budget_ratio if self.alpha is None else None
                ),
                split_name=split_name,
            )
            save_evaluation_artifacts(
                result,
                evaluation_dir,
                split_name=split_name,
                rolling_window=self.config.evaluation.rolling_risk_window,
                lambda_history=[0.0],
                lambda_update_steps=[1],
            )
            summaries[split_name] = result.summary
        summary = {
            "algo": self.algo,
            "seed": self.seed,
            "alpha": self.alpha,
            "alpha_budget_ratio": float(self.config.rcpo.alpha_budget_ratio),
            "lambda_lr_up": float(self.config.rcpo.lambda_lr_up),
            "lambda_lr_down": float(self.config.rcpo.lambda_lr_down),
            "reward_noise_enabled": bool(self.config.reward_noise.enabled),
            "reward_noise_mode": self.config.reward_noise.mode,
            "reward_noise_std": float(self.config.reward_noise.std),
            "constraint_mode": self.config.rcpo.constraint_mode,
            "drawdown_benchmark_mode": self.config.environment.drawdown_benchmark_mode,
            "action_mode": self.config.environment.action_mode,
            "simplex_action_format": self.config.environment.simplex_action_format,
            "policy_architecture": self.config.network.policy_architecture,
            "branch_credit_mode": self.config.network.branch_credit_mode,
            "initial_portfolio_mode": self.config.environment.initial_portfolio_mode,
            "reward_correction_mode": self.config.reward_correction.mode,
            "device": str(self.device),
            "constraint_preset": self._resolved_preset["preset_name"],
            "splits": summaries,
        }
        with (self.run_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary


def run_experiment(
    config: ProjectConfig,
    algo: str,
    output_root: str | Path | None = None,
    profiler: TrainingProfiler | None = None,
    disable_artifacts: bool = False,
    skip_validation: bool = False,
) -> list[Path]:
    if algo not in {"rcpo", "ppo_unconstrained", "equal_weight"}:
        raise ValueError(f"Unsupported algorithm: {algo}")
    if algo == "equal_weight" and config.reward_correction.mode != "none":
        raise ValueError("equal_weight does not support DRC/GDRC reward correction.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(output_root or config.experiment.output_root)
    reward_mode = config.reward_correction.mode
    run_root = base_root / f"{config.experiment.run_name}_{algo}_{reward_mode}_{timestamp}"
    run_directories: list[Path] = []
    for seed in config.experiment.seeds:
        run_dir = run_root / f"seed_{seed}"
        trainer = RCPOTrainer(
            config=config,
            algo=algo,
            run_dir=run_dir,
            seed=seed,
            profiler=profiler,
            disable_artifacts=disable_artifacts,
            skip_validation=skip_validation,
        )
        with exclusive_run_lock(run_dir):
            trainer.train()
        run_directories.append(run_dir)
    return run_directories


def resume_experiment(
    config: ProjectConfig,
    algo: str,
    run_dir: str | Path,
    checkpoint_name: str = "checkpoint_last.pt",
    profiler: TrainingProfiler | None = None,
    disable_artifacts: bool = False,
    skip_validation: bool = False,
) -> Path:
    run_path = Path(run_dir)
    checkpoint_path = run_path / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    seed = int(checkpoint.get("seed", config.experiment.seeds[0]))
    trainer = RCPOTrainer(
        config=config,
        algo=algo,
        run_dir=run_path,
        seed=seed,
        resume_checkpoint=checkpoint_path,
        profiler=profiler,
        disable_artifacts=disable_artifacts,
        skip_validation=skip_validation,
    )
    with exclusive_run_lock(run_path):
        trainer.train()
    return run_path
