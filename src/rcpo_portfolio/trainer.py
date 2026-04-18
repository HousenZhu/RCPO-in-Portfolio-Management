from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ProjectConfig, save_config, sync_rcpo_constraint_settings
from .env import PortfolioEnv
from .evaluation import evaluate_policy, save_evaluation_artifacts, save_training_progress_artifacts
from .market import generate_continuation_splits, generate_market_splits
from .models import ActorCritic


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor
    reward_returns: torch.Tensor
    cost_returns: torch.Tensor
    reward_advantages: torch.Tensor
    cost_advantages: torch.Tensor
    info_summary: dict[str, float]


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def combine_advantages(
    reward_advantages: torch.Tensor,
    cost_advantages: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    return reward_advantages - float(lambda_value) * cost_advantages


def update_lagrange_multiplier(
    lambda_value: float,
    observed_cost: float,
    alpha: float,
    learning_rate: float,
) -> float:
    return max(0.0, float(lambda_value + learning_rate * (observed_cost - alpha)))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=rewards.dtype)
    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[index]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[index]
            next_values = values[index + 1]
        delta = rewards[index] + gamma * next_values * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return advantages, returns


def _flatten_metrics(metrics: list[dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    return float(np.mean([metric[key] for metric in metrics]))


class RCPOTrainer:
    def __init__(
        self,
        config: ProjectConfig,
        algo: str,
        run_dir: Path,
        seed: int,
        resume_checkpoint: Path | None = None,
    ) -> None:
        self.config = config
        self.algo = algo
        self.run_dir = run_dir
        self.seed = seed
        self.resume_checkpoint = resume_checkpoint
        self.resume_completed_updates = 0
        set_global_seeds(seed)
        sync_rcpo_constraint_settings(config)

        market_splits = generate_market_splits(config.market, seed)
        self.train_market = market_splits["train"]
        self.train_env = PortfolioEnv(config.environment, market_splits["train"], config.market, seed=seed)
        self.validation_env = PortfolioEnv(
            config.environment, market_splits["validation"], config.market, seed=seed + 1
        )
        self.test_env = PortfolioEnv(config.environment, market_splits["test"], config.market, seed=seed + 2)
        self.optimization = config.ppo if algo == "ppo_unconstrained" else config.optimization

        obs_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.shape[0]
        self.model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, config=config.network)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimization.learning_rate)
        self.lambda_value = 0.0 if algo == "ppo_unconstrained" else config.rcpo.initial_lambda
        self.alpha = self._resolve_alpha()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.lambda_history: list[float] = [float(self.lambda_value)]
        self._resolved_preset = self.train_env.resolved_constraint_preset()
        if self.resume_checkpoint is not None:
            self._load_resume_checkpoint()

    def _print_run_header(self) -> None:
        total_updates = self.optimization.total_updates
        rollout_steps = self.optimization.rollout_steps
        print(
            f"[train] algo={self.algo} seed={self.seed} updates={total_updates} "
            f"rollout_steps={rollout_steps} alpha={self.alpha} "
            f"constraint_mode={self.config.rcpo.constraint_mode} "
            f"sortino_target={self.config.rcpo.sortino_target} "
            f"preset={self._resolved_preset['preset_name']} "
            f"group_a_min={self._resolved_preset['group_a_min_weight']} "
            f"group_b_max={self._resolved_preset['group_b_max_weight']} "
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
                f"batch_constraint={metric_row['batch_constraint_cost_mean']:.6f} "
                f"batch_downside={metric_row['batch_downside_cost_mean']:.6f} "
                f"batch_sortino_cost={metric_row['batch_sortino_violation_cost_mean']:.6f} "
                f"group_diag={metric_row['batch_group_a_min_violation_cost_mean'] + metric_row['batch_group_b_max_violation_cost_mean']:.6f} "
                f"episode_return={metric_row['episode_return_mean']:.6f} "
                f"turnover={metric_row['batch_turnover_mean']:.6f} "
                f"policy_loss={metric_row['policy_loss']:.6f} "
                f"value_loss_r={metric_row['reward_value_loss']:.6f} "
                f"value_loss_c={metric_row['cost_value_loss']:.6f} "
                f"lambda={metric_row['lambda_value']:.6f} "
                f"alpha={metric_row['alpha']} "
                f"kl={metric_row['approx_kl']:.6f} "
                f"clip_frac={metric_row['clip_fraction']:.4f} "
                f"lr={metric_row['learning_rate']:.8f} "
                f"val_return={metric_row['validation_annualized_return']:.6f} "
                f"val_sortino={metric_row['validation_sortino']:.6f} "
                f"val_constraint={metric_row['validation_constraint_cost']:.6f} "
                f"val_downside={metric_row['validation_downside_cost']:.8f} "
                f"val_sortino_cost={metric_row['validation_sortino_violation_cost']:.6f} "
                f"val_violation={metric_row['validation_constraint_violation_rate']:.2%} "
                f"score={validation_score:.6f}{best_marker}"
            ),
            flush=True,
        )

    def _equal_weight_action(self) -> np.ndarray:
        return np.full(
            self.train_env.action_space.shape[0],
            1.0 / self.train_env.num_assets,
            dtype=np.float32,
        )

    def _policy_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = self.model.get_action_and_value(
                observation_tensor, deterministic=deterministic
            )
        return action.squeeze(0).cpu().numpy()

    def _resolve_alpha(self) -> float | None:
        if self.algo == "ppo_unconstrained":
            return self.config.rcpo.alpha
        if self.config.rcpo.alpha is not None:
            return float(self.config.rcpo.alpha)
        if self.config.rcpo.constraint_mode == "sortino":
            return 0.0
        episodes = self.config.rcpo.calibration_episodes
        calibration = evaluate_policy(
            self.train_env,
            policy_fn=lambda _obs: self._equal_weight_action(),
            episodes=episodes,
            alpha=None,
            split_name="train_calibration",
        )
        return float(
            max(
                1e-6,
                calibration.summary["average_constraint_cost"] * self.config.rcpo.calibration_scale,
            )
        )

    def _collect_rollout(self) -> RolloutBatch:
        optimization = self.optimization
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        costs: list[float] = []
        dones: list[float] = []
        reward_values: list[float] = []
        cost_values: list[float] = []
        episode_metrics: list[dict[str, float]] = []
        downside_costs: list[float] = []
        normalized_downside_costs: list[float] = []
        sortino_violation_costs: list[float] = []
        sortino_ratios: list[float] = []
        group_a_min_violation_costs: list[float] = []
        group_b_max_violation_costs: list[float] = []
        group_a_weights: list[float] = []
        group_b_weights: list[float] = []

        obs, _ = self.train_env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_turnover = 0.0
        episode_steps = 0
        for _ in range(optimization.rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob_tensor, _, reward_value_tensor, cost_value_tensor = (
                    self.model.get_action_and_value(obs_tensor)
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            observations.append(obs.astype(np.float32))
            actions.append(action.astype(np.float32))
            log_probs.append(float(log_prob_tensor.item()))
            rewards.append(float(reward))
            costs.append(float(info["constraint_cost"]))
            dones.append(float(done))
            reward_values.append(float(reward_value_tensor.item()))
            cost_values.append(float(cost_value_tensor.item()))
            downside_costs.append(float(info["downside_cost"]))
            normalized_downside_costs.append(float(info["normalized_downside_cost"]))
            sortino_violation_costs.append(float(info["sortino_violation_cost"]))
            sortino_ratios.append(float(info["sortino_ratio"]))
            group_a_min_violation_costs.append(float(info["group_a_min_violation_cost"]))
            group_b_max_violation_costs.append(float(info["group_b_max_violation_cost"]))
            group_a_weights.append(float(info["group_a_weight"]))
            group_b_weights.append(float(info["group_b_weight"]))

            episode_reward += float(info["net_return"])
            episode_cost += float(info["constraint_cost"])
            episode_turnover += float(info["turnover"])
            episode_steps += 1

            obs = next_obs
            if done:
                episode_metrics.append(
                    {
                        "episode_return": episode_reward,
                        "episode_cost": episode_cost / max(episode_steps, 1),
                        "episode_turnover": episode_turnover / max(episode_steps, 1),
                    }
                )
                obs, _ = self.train_env.reset()
                episode_reward = 0.0
                episode_cost = 0.0
                episode_turnover = 0.0
                episode_steps = 0

        next_value_r, next_value_c = self.model.value(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        )
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        costs_tensor = torch.as_tensor(costs, dtype=torch.float32)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32)
        reward_values_tensor = torch.as_tensor(reward_values, dtype=torch.float32)
        cost_values_tensor = torch.as_tensor(cost_values, dtype=torch.float32)
        reward_advantages, reward_returns = compute_gae(
            rewards_tensor,
            reward_values_tensor,
            dones_tensor,
            next_value_r.squeeze(0).detach(),
            optimization.gamma,
            optimization.gae_lambda,
        )
        cost_advantages, cost_returns = compute_gae(
            costs_tensor,
            cost_values_tensor,
            dones_tensor,
            next_value_c.squeeze(0).detach(),
            optimization.gamma,
            optimization.gae_lambda,
        )
        return RolloutBatch(
            observations=torch.as_tensor(np.asarray(observations), dtype=torch.float32),
            actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32),
            log_probs=torch.as_tensor(log_probs, dtype=torch.float32),
            rewards=rewards_tensor,
            costs=costs_tensor,
            dones=dones_tensor,
            reward_values=reward_values_tensor,
            cost_values=cost_values_tensor,
            reward_returns=reward_returns.detach(),
            cost_returns=cost_returns.detach(),
            reward_advantages=reward_advantages.detach(),
            cost_advantages=cost_advantages.detach(),
            info_summary={
                "batch_reward_mean": float(np.mean(rewards)),
                "batch_constraint_cost_mean": float(np.mean(costs)),
                "batch_downside_cost_mean": float(np.mean(downside_costs)),
                "batch_normalized_downside_cost_mean": float(np.mean(normalized_downside_costs)),
                "batch_sortino_violation_cost_mean": float(np.mean(sortino_violation_costs)),
                "batch_sortino_ratio_mean": float(np.mean(sortino_ratios)),
                "batch_group_a_min_violation_cost_mean": float(np.mean(group_a_min_violation_costs)),
                "batch_group_b_max_violation_cost_mean": float(np.mean(group_b_max_violation_costs)),
                "batch_group_a_weight_mean": float(np.mean(group_a_weights)),
                "batch_group_b_weight_mean": float(np.mean(group_b_weights)),
                "batch_turnover_mean": _flatten_metrics(episode_metrics, "episode_turnover"),
                "episode_return_mean": _flatten_metrics(episode_metrics, "episode_return"),
                "episode_cost_mean": _flatten_metrics(episode_metrics, "episode_cost"),
            },
        )

    def _update_model(self, batch: RolloutBatch) -> dict[str, float]:
        optimization = self.optimization
        combined_advantages = combine_advantages(
            batch.reward_advantages, batch.cost_advantages, self.lambda_value if self.algo == "rcpo" else 0.0
        )
        combined_advantages = (combined_advantages - combined_advantages.mean()) / (
            combined_advantages.std(unbiased=False) + 1e-8
        )
        reward_advantages = (batch.reward_advantages - batch.reward_advantages.mean()) / (
            batch.reward_advantages.std(unbiased=False) + 1e-8
        )
        batch_size = batch.observations.shape[0]
        policy_losses: list[float] = []
        reward_value_losses: list[float] = []
        cost_value_losses: list[float] = []
        entropy_terms: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        stopped_by_target_kl = False

        for _ in range(optimization.epochs):
            permutation = torch.randperm(batch_size)
            for start in range(0, batch_size, optimization.minibatch_size):
                batch_indices = permutation[start : start + optimization.minibatch_size]
                observations = batch.observations[batch_indices]
                actions = batch.actions[batch_indices]
                old_log_probs = batch.log_probs[batch_indices]
                reward_returns = batch.reward_returns[batch_indices]
                cost_returns = batch.cost_returns[batch_indices]

                selected_advantages = (
                    reward_advantages[batch_indices]
                    if self.algo == "ppo_unconstrained"
                    else combined_advantages[batch_indices]
                )
                _, new_log_probs, entropy, reward_values, cost_values = self.model.get_action_and_value(
                    observations, action=actions
                )
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - optimization.clip_epsilon, 1.0 + optimization.clip_epsilon
                )
                with torch.no_grad():
                    approx_kl = torch.mean(old_log_probs - new_log_probs)
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > optimization.clip_epsilon).float()
                    )
                policy_loss = -torch.mean(
                    torch.min(ratio * selected_advantages, clipped_ratio * selected_advantages)
                )
                reward_value_loss = torch.mean(torch.square(reward_returns - reward_values))
                cost_value_loss = torch.mean(torch.square(cost_returns - cost_values))
                entropy_bonus = torch.mean(entropy)

                total_loss = (
                    policy_loss
                    + optimization.reward_value_coef * reward_value_loss
                    + optimization.cost_value_coef * cost_value_loss
                    - optimization.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), optimization.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                reward_value_losses.append(float(reward_value_loss.item()))
                cost_value_losses.append(float(cost_value_loss.item()))
                entropy_terms.append(float(entropy_bonus.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

                if (
                    self.algo == "ppo_unconstrained"
                    and optimization.target_kl is not None
                    and float(approx_kl.item()) > optimization.target_kl
                ):
                    stopped_by_target_kl = True
                    break

            if stopped_by_target_kl:
                break

            if self.algo == "rcpo" and self.alpha is not None:
                self.lambda_value = update_lagrange_multiplier(
                    self.lambda_value,
                    batch.info_summary["batch_constraint_cost_mean"],
                    self.alpha,
                    self.config.rcpo.lambda_lr,
                )
                self.lambda_history.append(float(self.lambda_value))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "reward_value_loss": float(np.mean(reward_value_losses)),
            "cost_value_loss": float(np.mean(cost_value_losses)),
            "entropy": float(np.mean(entropy_terms)),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "ppo_kl_early_stop": float(stopped_by_target_kl),
            "combined_advantage_mean": float(combined_advantages.mean().item()),
        }

    def _write_metric_row(self, payload: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _checkpoint(self, name: str) -> None:
        torch.save(
            {
                "algo": self.algo,
                "seed": self.seed,
                "alpha": self.alpha,
                "constraint_mode": self.config.rcpo.constraint_mode,
                "lambda_value": self.lambda_value,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "completed_updates": self.resume_completed_updates,
            },
            self.run_dir / name,
        )

    def _load_resume_checkpoint(self) -> None:
        if self.resume_checkpoint is None:
            return
        checkpoint = torch.load(self.resume_checkpoint, map_location="cpu")
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
        checkpoint_constraint_mode = checkpoint.get("constraint_mode")
        if (
            self.algo == "rcpo"
            and checkpoint_constraint_mode is not None
            and checkpoint_constraint_mode != self.config.rcpo.constraint_mode
        ):
            raise ValueError(
                f"Checkpoint constraint mode {checkpoint_constraint_mode!r} does not match "
                f"requested {self.config.rcpo.constraint_mode!r}."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(
                "[train] resume_warning=checkpoint_has_no_optimizer_state using_fresh_optimizer=true",
                flush=True,
            )
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
                "annualized_return": row["validation_annualized_return"],
                "sortino": row["validation_sortino"],
                "average_constraint_cost": row["validation_constraint_cost"],
                "average_downside_cost": row["validation_downside_cost"],
                "average_sortino_violation_cost": row.get(
                    "validation_sortino_violation_cost", 0.0
                ),
                "average_step_sortino_ratio": row.get("validation_step_sortino_ratio", 0.0),
                "max_drawdown": row.get("validation_max_drawdown", 0.0),
                "average_group_a_weight": row.get("validation_group_a_weight", 0.0),
                "average_group_b_weight": row.get("validation_group_b_weight", 0.0),
                "average_group_a_min_violation_cost": row.get(
                    "validation_group_a_min_violation_cost", 0.0
                ),
                "average_group_b_max_violation_cost": row.get(
                    "validation_group_b_max_violation_cost", 0.0
                ),
                "constraint_violation_rate": row.get("validation_constraint_violation_rate", 0.0),
                "split": "validation",
            }
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
        return float(validation_summary["annualized_return"])

    def _evaluate_checkpoint_artifacts(
        self,
        checkpoint_name: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        checkpoint = torch.load(self.run_dir / checkpoint_name, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        summaries: dict[str, Any] = {}
        for split_name, env in [
            ("validation", self.validation_env),
            ("test", self.test_env),
        ]:
            result = evaluate_policy(
                env,
                policy_fn=lambda obs: self._policy_action(obs, deterministic=True),
                episodes=self.config.evaluation.episodes,
                alpha=self.alpha,
                split_name=split_name,
            )
            mean_model_returns, mean_equal_weight_returns = self._future_branch_returns(
                split_name=split_name,
                policy_fn=lambda obs: self._policy_action(obs, deterministic=True),
            )
            save_evaluation_artifacts(
                result,
                output_dir,
                split_name=split_name,
                rolling_window=self.config.evaluation.rolling_risk_window,
                lambda_history=self.lambda_history,
                mean_episode_returns=mean_model_returns,
                equal_weight_mean_episode_returns=mean_equal_weight_returns,
            )
            summaries[split_name] = result.summary
        return summaries

    def _future_branch_returns(
        self,
        split_name: str,
        policy_fn,
    ) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
        if split_name not in {"validation", "test"}:
            return None, None
        steps = (
            self.config.market.validation_steps
            if split_name == "validation"
            else self.config.market.test_steps
        )
        seed_offset = 10_000 if split_name == "validation" else 20_000
        future_markets = generate_continuation_splits(
            self.config.market,
            self.train_market,
            steps,
            self.seed + seed_offset,
            count=5,
        )
        model_returns: list[np.ndarray] = []
        equal_weight_returns: list[np.ndarray] = []
        for index, market in enumerate(future_markets):
            env = PortfolioEnv(
                self.config.environment,
                market,
                self.config.market,
                seed=self.seed + seed_offset + index,
            )
            start_index = int(env.available_start_indices()[0])
            model_result = evaluate_policy(
                env,
                policy_fn=policy_fn,
                episodes=1,
                alpha=self.alpha,
                split_name=split_name,
            )
            model_returns.append(model_result.first_episode["net_returns"])
            env_for_equal_weight = PortfolioEnv(
                self.config.environment,
                market,
                self.config.market,
                seed=self.seed + seed_offset + index,
            )
            equal_weight_obs, _ = env_for_equal_weight.reset(options={"start_index": start_index})
            branch_returns: list[float] = []
            while True:
                action = self._equal_weight_action()
                equal_weight_obs, reward, terminated, truncated, info = env_for_equal_weight.step(action)
                del reward
                branch_returns.append(float(info["net_return"]))
                if terminated or truncated:
                    break
            equal_weight_returns.append(np.asarray(branch_returns, dtype=np.float32))
        return model_returns, equal_weight_returns

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config.environment.resolved_group_a_min_weight = float(
            self._resolved_preset["group_a_min_weight"]
        )
        self.config.environment.resolved_group_b_max_weight = float(
            self._resolved_preset["group_b_max_weight"]
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
        start_update = max(
            self.resume_completed_updates,
            len(metrics_rows),
        )
        target_total_updates = start_update + additional_updates
        if metrics_rows:
            self.lambda_history = [
                float(row.get("lambda_value", self.lambda_value)) for row in metrics_rows
            ]
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
            current_learning_rate = self._set_learning_rate(update_index, target_total_updates)
            rollout = self._collect_rollout()
            losses = self._update_model(rollout)
            validation_result = evaluate_policy(
                self.validation_env,
                policy_fn=lambda obs: self._policy_action(obs, deterministic=True),
                episodes=self.config.evaluation.episodes,
                alpha=self.alpha,
                split_name="validation",
            )
            validation_score = self._validation_score(validation_result.summary)
            metric_row = {
                "update": update_index,
                "algo": self.algo,
                "alpha": self.alpha,
                "constraint_mode": self.config.rcpo.constraint_mode,
                "lambda_value": self.lambda_value,
                "learning_rate": current_learning_rate,
                "constraint_preset": self._resolved_preset["preset_name"],
                "turnover_cap": self.config.environment.turnover_cap,
                **rollout.info_summary,
                **losses,
                "validation_annualized_return": validation_result.summary["annualized_return"],
                "validation_sortino": validation_result.summary["sortino"],
                "validation_max_drawdown": validation_result.summary["max_drawdown"],
                "validation_turnover": validation_result.summary["average_turnover"],
                "validation_constraint_cost": validation_result.summary["average_constraint_cost"],
                "validation_downside_cost": validation_result.summary["average_downside_cost"],
                "validation_sortino_violation_cost": validation_result.summary["average_sortino_violation_cost"],
                "validation_step_sortino_ratio": validation_result.summary["average_step_sortino_ratio"],
                "validation_group_a_weight": validation_result.summary["average_group_a_weight"],
                "validation_group_b_weight": validation_result.summary["average_group_b_weight"],
                "validation_group_a_min_violation_cost": validation_result.summary["average_group_a_min_violation_cost"],
                "validation_group_b_max_violation_cost": validation_result.summary["average_group_b_max_violation_cost"],
                "validation_constraint_violation_rate": validation_result.summary["constraint_violation_rate"],
            }
            self._write_metric_row(metric_row)
            metrics_rows.append(metric_row)
            is_best = False
            previous_best_score = best_score
            min_delta = (
                self.optimization.early_stop_min_delta
                if self.algo == "ppo_unconstrained"
                else 0.0
            )
            if validation_score > best_score:
                best_score = validation_score
                best_summary = validation_result.summary
                self._checkpoint("checkpoint_best.pt")
                is_best = True
            if validation_score > previous_best_score + min_delta:
                stale_updates = 0
            else:
                stale_updates += 1
            self.resume_completed_updates = update_index + 1
            self._checkpoint("checkpoint_last.pt")
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
                self.algo == "ppo_unconstrained"
                and self.optimization.early_stop_patience is not None
                and stale_updates >= self.optimization.early_stop_patience
            ):
                early_stopped = True
                print(
                    f"[train] early_stop=validation_return patience={self.optimization.early_stop_patience} "
                    f"completed_updates={self.resume_completed_updates}",
                    flush=True,
                )
                break

        if best_summary is None:
            raise RuntimeError("Training did not produce a validation summary.")
        save_training_progress_artifacts(metrics_rows, self.run_dir / "evaluation")
        best_evaluation = self._evaluate_checkpoint_artifacts(
            "checkpoint_best.pt",
            self.run_dir / "evaluation_best",
        )
        last_evaluation = self._evaluate_checkpoint_artifacts(
            "checkpoint_last.pt",
            self.run_dir / "evaluation_last",
        )
        summary = {
            "algo": self.algo,
            "seed": self.seed,
            "alpha": self.alpha,
            "constraint_mode": self.config.rcpo.constraint_mode,
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
                split_name=split_name,
            )
            save_evaluation_artifacts(
                result,
                evaluation_dir,
                split_name=split_name,
                rolling_window=self.config.evaluation.rolling_risk_window,
                lambda_history=[0.0],
            )
            summaries[split_name] = result.summary
        summary = {
            "algo": self.algo,
            "seed": self.seed,
            "alpha": self.alpha,
            "constraint_mode": self.config.rcpo.constraint_mode,
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
) -> list[Path]:
    if algo not in {"rcpo", "ppo_unconstrained", "equal_weight"}:
        raise ValueError(f"Unsupported algorithm: {algo}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(output_root or config.experiment.output_root)
    run_root = base_root / f"{config.experiment.run_name}_{algo}_{timestamp}"
    run_directories: list[Path] = []
    for seed in config.experiment.seeds:
        run_dir = run_root / f"seed_{seed}"
        trainer = RCPOTrainer(config=config, algo=algo, run_dir=run_dir, seed=seed)
        trainer.train()
        run_directories.append(run_dir)
    return run_directories


def resume_experiment(
    config: ProjectConfig,
    algo: str,
    run_dir: str | Path,
    checkpoint_name: str = "checkpoint_last.pt",
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
    )
    trainer.train()
    return run_path
