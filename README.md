# RCPO Portfolio Management Prototype

This project builds a synthetic portfolio management problem and solves it with a reward-constrained PPO variant inspired by Reward Constrained Policy Optimization (RCPO).

## Features

- Synthetic 2-regime market with 5 risky assets plus cash
- Gymnasium-compatible portfolio environment
- Long-only portfolio weights via softmax-transformed policy logits
- RCPO-style constrained optimization with separate reward and cost critics
- Baselines for unconstrained PPO and equal-weight allocation
- Config-driven training, evaluation, metrics, checkpoints, and plots

## Quickstart

```powershell
python -m pip install -e .[dev]
python train.py --algo rcpo --constraint-downside --config configs/default.yaml
python train.py --algo rcpo --constraint-sortino --config configs/default.yaml
python train.py --algo ppo_unconstrained --config configs/default.yaml
python evaluate.py --run-dir runs/latest_rcpo_seed0
pytest
```

## Constraint Definition

- Reward: net portfolio log return after transaction costs
- `--constraint-downside`: cost is normalized downside semivariance `max(0, -r_t)^2 / downside_cost_scale`
- `--constraint-sortino`: cost is a target-violation penalty `max(0, sortino_target - rolling_sortino)^2 / sortino_cost_scale`
- Group allocation bounds remain available in config and reports as diagnostics only; they do not affect RCPO training.

## Outputs

Training creates a run directory containing:

- `config_snapshot.yaml`
- `metrics.jsonl`
- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `evaluation/summary_*.json`
- `evaluation/*.png`
