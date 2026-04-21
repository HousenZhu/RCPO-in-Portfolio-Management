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
python train.py --algo ppo_unconstrained --use-drc --config configs/default.yaml
python train.py --algo ppo_unconstrained --use-gdrc --config configs/default.yaml
python train.py --algo rcpo --constraint-downside --use-drc --config configs/default.yaml
python train.py --algo rcpo --constraint-downside --use-gdrc --config configs/default.yaml
python train.py --algo rcpo --constraint-sortino --use-gdrc --config configs/default.yaml
python evaluate.py --run-dir runs/latest_rcpo_seed0
pytest
```
py -3.11 train.py --algo ppo_unconstrained --config configs/default.yaml

py -3.11 train.py --algo rcpo --constraint-downside --use-gdrc --config configs/default.yaml
py -3.11 train.py --algo rcpo --constraint-downside --resume-run-dir "runs\new_rcpo_none_20260420_201503\seed_0"
py -3.11 train.py --algo ppo_unconstrained --resume-run-dir "runs\new_ppo_unconstrained_none_20260420_195339\seed_0"

py -3.11 evaluate.py --run-dir "runs\latest_ppo_unconstrained_20260416_220423\seed_0" --checkpoint checkpoint_last.pt --future-market-count 10
py -3.11 evaluate.py --run-dir "runs\latest_rcpo_20260417_183003\seed_0" --checkpoint checkpoint_last.pt --include-train-seed-future --train-seed-future-steps 252

latest_rcpo_20260417_183003
latest_rcpo_gdrc_20260419_173123
latest_ppo_unconstrained_20260416_220423
new_rcpo_none_20260420_201503
new_ppo_unconstrained_none_20260420_195339

Train on many markets, not one train path. Select checkpoints by mean validation over many branches. Increase train length to 5040. Raise transaction_cost_bps to 1.0. Use equal weight as a prior - start near equal weight, then let the policy learn small tilts. Modify the market to show RL can beat equal weight - Add learnable structure: regime-dependent asset winners, momentum, mean reversion, volatility forecasting, or changing correlations - analyze these market modification and choose which to implement. Add diversification regularization - as an RCPO cost:

constraint_cost = downside_cost + beta * concentration_cost; concentration = sum(weights^2). 

i am required to explain the whole project to my professor. summarize environement, algorithm and results so far from below files:
latest_rcpo_none_20260419_173124, latest_rcpo_20260417_183003, latest_rcpo_gdrc_20260419_173123, latest_ppo_unconstrained_20260416_220423.
generate detailed report and leave the place for proper figures

为啥turnover一开始0后面涨起来
  synthetic two-regime market and long-only portfolio weights. 
 
  downside and Sortino constraint modes

  optional DRC/GDRC reward correction

  robust multi-market design intended to reduce train-path overfitting.

The central empirical finding so far is that PPO and RCPO can learn high-return policies on the training paths, but early versions often overfit a small number of synthetic market paths. RCPO variants improved validation return but also produced very high turnover. GDRC did not fully solve something. 

current robust upgrade: 
train on many markets, 
select checkpoints using mean validation performance over multiple future branches, 
initialize near equal weight, 
add learnable market structure


## Constraint Definition

- Reward: net portfolio log return after transaction costs
- `--constraint-downside`: cost is normalized downside semivariance `max(0, -r_t)^2 / downside_cost_scale`
- `--constraint-sortino`: cost is a target-violation penalty `max(0, sortino_target - rolling_sortino)^2 / sortino_cost_scale`
- Group allocation bounds remain available in config and reports as diagnostics only; they do not affect RCPO training.

## DRC / GDRC Reward Correction

Optional reward-correction modes are available for both PPO and RCPO:

- No flag: train on the environment reward directly.
- `--use-drc`: train a single distributional reward critic and use its corrected reward for reward advantages/value targets.
- `--use-gdrc`: train an ensemble of DRC critics and select the active critic using ordinal-cross-entropy behavior.

DRC/GDRC only change the reward stream used for training. Evaluation summaries and plots still report actual portfolio returns from the environment.

The default GDRC settings are tuned for the current synthetic portfolio reward scale: six candidate critics use bin counts `2, 4, 6, 8, 10, 12`; the adaptive reward range uses recent rewards between the `0.5` and `99.5` percentiles; and each corrector trains for `3` epochs per rollout update to reduce rollout memorization.

## Device Selection

Training selects the PyTorch device from YAML only. The default is `runtime.device: auto`, which uses CUDA when PyTorch can see a GPU and otherwise falls back to CPU. Set `runtime.device: cpu` in the config to force CPU.

## Outputs

Training creates a run directory containing:

- `config_snapshot.yaml`
- `metrics.jsonl`
- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `evaluation/summary_*.json`
- `evaluation/*.png`
