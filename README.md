# RCPO Portfolio Management Prototype

This project builds a synthetic portfolio management problem and trains policy-gradient agents with PyTorch and Gymnasium. The main methods are unconstrained PPO and an RCPO-style PPO variant that maximizes net portfolio log return while enforcing a benchmark-relative maximum drawdown constraint.

## Features

- Synthetic multi-market generator with 5 risky assets plus cash
- Long-only portfolio weights via softmax-transformed policy logits
- PPO baseline and RCPO with a reward critic, cost critic, and Lagrange multiplier
- Benchmark-relative maximum drawdown constraint for RCPO
- Optional DRC/GDRC reward correction for PPO or RCPO
- Config-driven training, resume, evaluation, checkpoints, metrics, and plots

## Install

```powershell
py -3.11 -m pip install -e .[dev]
```

## Commands

Train PPO:

```powershell
py -3.11 train.py --algo ppo_unconstrained --config configs/default.yaml
```

Train PPO with reward correction:

```powershell
py -3.11 train.py --algo ppo_unconstrained --use-drc --config configs/default.yaml
py -3.11 train.py --algo ppo_unconstrained --use-gdrc --config configs/default.yaml
```

Train RCPO with the maximum drawdown constraint:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/default.yaml
```

Train RCPO with reward correction:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --use-drc --config configs/default.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --use-gdrc --config configs/default.yaml
```

Train with Gaussian reward noise when `reward_noise.enabled: true` in `configs/default.yaml`:

```powershell
py -3.11 train.py --algo ppo_unconstrained --config configs/default.yaml
py -3.11 train.py --algo ppo_unconstrained --use-gdrc --config configs/default.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/default.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --use-gdrc --config configs/default.yaml
```

Run the equal-weight baseline:

```powershell
py -3.11 train.py --algo equal_weight --config configs/default.yaml
```

Resume RCPO from the last checkpoint:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --resume-run-dir "runs\new_rcpo_none_20260427_184555\seed_0"
```

Resume RCPO with GDRC:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --use-gdrc --resume-run-dir "runs\noise_v1_rcpo_gdrc_20260429_121506\seed_0"
```

Resume PPO:

```powershell
py -3.11 train.py --algo ppo_unconstrained  --resume-run-dir "runs\new_ppo_unconstrained_none_20260424_170155\seed_0"
```

To resume from a specific checkpoint file inside the run directory:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --resume-run-dir "runs\new_rcpo_none_YYYYMMDD_HHMMSS\seed_0" --resume-checkpoint checkpoint_best.pt
```
noise_v1_ppo_unconstrained_gdrc_20260429_153756

## Evaluation Commands

Evaluate the best checkpoint:

```powershell
py -3.11 evaluate.py --run-dir "runs\latest_rcpo_none_YYYYMMDD_HHMMSS\seed_0" --checkpoint checkpoint_best.pt
```

Evaluate the last checkpoint with 20 future continuation markets:

```powershell
py -3.11 evaluate.py --run-dir "runs\new_ppo_unconstrained_none_20260424_170155\seed_0" --checkpoint checkpoint_last.pt --future-market-count 20
```

Evaluate one future market that uses the same numeric seed as the train split:

```powershell
py -3.11 evaluate.py --run-dir "runs\latest_rcpo_none_YYYYMMDD_HHMMSS\seed_0" --checkpoint checkpoint_last.pt --include-train-seed-future --train-seed-future-steps 252
```

## Constraint Definition

Reward is net portfolio log return after transaction costs:

```text
reward_t = log(1 + net_simple_return_t)
```

RCPO uses a benchmark-relative maximum drawdown constraint. During each episode, the environment tracks the agent portfolio path and an online equal-weight benchmark path under the same transaction-cost model:

```text
agent_current_drawdown_t = (agent_running_peak_t - agent_portfolio_value_t) / agent_running_peak_t
agent_max_drawdown_t = max(previous_agent_max_drawdown, agent_current_drawdown_t)
equal_weight_max_drawdown_t = max(previous_equal_weight_max_drawdown, equal_weight_current_drawdown_t)
budget_t = max(drawdown_budget_floor, benchmark_drawdown_margin * equal_weight_max_drawdown_t)
drawdown_violation_t = max(0, agent_max_drawdown_t - budget_t)
constraint_cost = drawdown_violation^2 / drawdown_cost_scale
```

Default settings:

```yaml
environment:
  drawdown_budget_floor: 0.02
  benchmark_drawdown_margin: 0.90
  drawdown_cost_scale: 0.01

rcpo:
  alpha: null
  alpha_budget_ratio: 0.05
```

This means the policy can seek return, but RCPO penalizes episode paths whose running maximum drawdown exceeds a budget defined online from equal weight. The `0.90` margin asks the policy to stay about 10% safer than equal weight on drawdown, subject to the `0.02` minimum floor. With `alpha_budget_ratio: 0.05`, the Lagrange multiplier tolerates about 5% of the current effective drawdown budget as average violation before it increases.

## DRC / GDRC Reward Correction

Optional reward-correction modes are available for PPO and RCPO:

- No flag: train on the environment reward directly.
- `--use-drc`: train a single distributional reward critic and use corrected rewards for reward advantages/value targets.
- `--use-gdrc`: train a fine-bin GDRC ensemble and select between 48-bin and 64-bin reward critics.

DRC/GDRC only change the reward stream used for training. Evaluation summaries and plots still report actual portfolio returns from the environment.

The current GDRC stabilization settings are:

```yaml
reward_correction:
  mode: gdrc
  num_bins: 48
  gdrc_candidate_bins: [48, 64]
  correction_coef: 0.50
  correction_delta_clip: 0.0015
```

`correction_coef` applies only part of the critic's suggested reward correction, and `correction_delta_clip` caps the final per-step correction used for training.

## Reward Noise

`configs/default.yaml` includes an optional Gaussian training reward-noise channel. The current experiment config is set to `enabled: true`; set it to `false` for clean-reward training.

```yaml
reward_noise:
  enabled: true
  mode: gaussian
  std: 0.003
  seed_offset: 30000
```

With `enabled: true`, rollout collection trains on noisy observed rewards:

```text
observed_reward_t = true_reward_t + Normal(0, 0.003)
```

Only rollout training rewards are corrupted. Validation, test evaluation, checkpoint scoring, drawdown costs, and equal-weight comparisons continue to use clean portfolio returns.

## Device Selection

Training selects the PyTorch device from YAML only:

```yaml
runtime:
  device: auto
```

`auto` uses CUDA when PyTorch can see a GPU and otherwise falls back to CPU. Set `runtime.device: cpu` to force CPU.

## Outputs

Training creates a run directory containing:

- `config_snapshot.yaml`
- `metrics.jsonl`
- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `training_summary.json`
- `evaluation/`
- `evaluation_best/`
- `evaluation_last/`

Evaluation folders contain JSON summaries and PNG plots for cumulative return, mean cumulative return across future branches, portfolio weights, turnover, drawdown, drawdown constraint cost, and lambda trajectory.

## Tests

```powershell
py -3.11 -m pytest
```
