from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rcpo_portfolio.evaluation import (
    evaluate_policy,
    load_checkpoint_for_evaluation,
    save_evaluation_artifacts,
)
from rcpo_portfolio.env import PortfolioEnv
from rcpo_portfolio.market import generate_continuation_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RCPO portfolio run.")
    parser.add_argument("--run-dir", required=True, help="Run directory produced by train.py.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pt",
        help="Checkpoint file name inside the run directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional evaluation artifact directory. Defaults to evaluation_best for "
            "checkpoint_best.pt, evaluation_last for checkpoint_last.pt, otherwise evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, metadata, model, environments = load_checkpoint_for_evaluation(
        args.run_dir, checkpoint_name=args.checkpoint
    )
    lambda_history: list[float] | None = None
    metrics_path = Path(args.run_dir) / "metrics.jsonl"
    if metrics_path.exists():
        lambda_history = []
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                lambda_history.append(float(payload.get("lambda_value", 0.0)))

    def policy_fn(observation):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = model.get_action_and_value(
                observation_tensor, deterministic=config.evaluation.deterministic
            )
        return action.squeeze(0).cpu().numpy()

    def equal_weight_action(env: PortfolioEnv):
        return np.zeros(env.action_space.shape[0], dtype=np.float32)

    def rollout_returns(env: PortfolioEnv, action_fn):
        obs, _ = env.reset(options={"start_index": int(env.available_start_indices()[0])})
        returns = []
        while True:
            obs, reward, terminated, truncated, info = env.step(action_fn(obs))
            del reward
            returns.append(float(info["net_return"]))
            if terminated or truncated:
                break
        return np.asarray(returns, dtype=np.float32)

    def future_branch_returns(split_name: str):
        if split_name not in {"validation", "test"}:
            return None, None
        steps = (
            config.market.validation_steps
            if split_name == "validation"
            else config.market.test_steps
        )
        seed_offset = 10_000 if split_name == "validation" else 20_000
        future_markets = generate_continuation_splits(
            config.market,
            environments["train"].market,
            steps,
            metadata["seed"] + seed_offset,
            count=5,
        )
        model_returns = []
        equal_weight_returns = []
        for index, market in enumerate(future_markets):
            branch_env = PortfolioEnv(
                config.environment,
                market,
                config.market,
                seed=metadata["seed"] + seed_offset + index,
            )
            model_returns.append(rollout_returns(branch_env, policy_fn))
            equal_weight_env = PortfolioEnv(
                config.environment,
                market,
                config.market,
                seed=metadata["seed"] + seed_offset + index,
            )
            equal_weight_returns.append(
                rollout_returns(equal_weight_env, lambda _obs, env=equal_weight_env: equal_weight_action(env))
            )
        return model_returns, equal_weight_returns

    if args.output_dir is not None:
        evaluation_dir = Path(args.output_dir)
    elif args.checkpoint == "checkpoint_best.pt":
        evaluation_dir = Path(args.run_dir) / "evaluation_best"
    elif args.checkpoint == "checkpoint_last.pt":
        evaluation_dir = Path(args.run_dir) / "evaluation_last"
    else:
        evaluation_dir = Path(args.run_dir) / "evaluation"
    for split_name, env in environments.items():
        if split_name == "train":
            continue
        result = evaluate_policy(
            env,
            policy_fn=policy_fn,
            episodes=config.evaluation.episodes,
            alpha=metadata["alpha"],
            split_name=split_name,
        )
        model_mean_returns, equal_weight_mean_returns = future_branch_returns(split_name)
        save_evaluation_artifacts(
            result,
            evaluation_dir,
            split_name=split_name,
            rolling_window=config.evaluation.rolling_risk_window,
            lambda_history=lambda_history,
            mean_episode_returns=model_mean_returns,
            equal_weight_mean_episode_returns=equal_weight_mean_returns,
        )
        print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
