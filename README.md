# RCPO Portfolio Management Prototype

This project builds a synthetic portfolio management problem and trains policy-gradient agents with PyTorch and Gymnasium. The main methods are unconstrained PPO, simplex-decomposition policies with hard allocation feasibility, and RCPO-style PPO variants that maximize net portfolio log return while controlling either drawdown or soft allocation-constraint violations.

## Features

- Synthetic multi-market generator with 5 risky assets plus cash
- Long-only portfolio weights via softmax logits or CAOSD-style simplex decomposition
- PPO baseline and RCPO with a reward critic, cost critic, and Lagrange multiplier
- Benchmark-relative maximum drawdown constraint for RCPO
- Soft allocation-constraint RCPO baseline for comparison with hard simplex feasibility
- Combined allocation-plus-drawdown RCPO baseline for joint feasibility and risk control
- Two hard allocation constraints under `environment.action_mode: simplex_decomposition`
- Configurable CAOSD policies: flat Gaussian, parallel/autoregressive Gaussian logits, or autoregressive Dirichlet weights
- Global or standalone branch credit assignment for simplex policies
- Optional DRC/GDRC reward correction for PPO or RCPO
- Config-driven training, resume, evaluation, checkpoints, metrics, and plots

## Install

```powershell
py -3.11 -m pip install -e .[dev]
```

## Commands

Recommended six-experiment training set:

```powershell
py -3.11 train.py --algo ppo_unconstrained --config configs/simplex_ppo_gaussian.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/simplex_rcpo_gaussian.yaml
py -3.11 train.py --algo ppo_unconstrained --config configs/simplex_ppo_dirichlet.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/simplex_rcpo_dirichlet.yaml
py -3.11 train.py --algo rcpo --constraint-allocation --config configs/rcpo_allocation_penalty.yaml
py -3.11 train.py --algo rcpo --constraint-allocation-drawdown --config configs/rcpo_allocation_drawdown_penalty.yaml
```

Train PPO with the general default config:

```powershell
py -3.11 train.py --algo ppo_unconstrained --config configs/default.yaml
```

Train PPO with reward correction:

```powershell
py -3.11 train.py --algo ppo_unconstrained --use-drc --config configs/default.yaml
py -3.11 train.py --algo ppo_unconstrained --use-gdrc --config configs/default.yaml
```

Set `network.branch_credit_mode: global` before using DRC/GDRC or reward noise. Standalone branch credit currently requires clean rewards.

Train RCPO with the maximum drawdown constraint:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/default.yaml
```

Train the soft allocation-constraint RCPO baseline without simplex decomposition:

```powershell
py -3.11 train.py --algo rcpo --constraint-allocation --config configs/rcpo_allocation_penalty.yaml
```

Train the joint soft allocation-plus-drawdown RCPO baseline:

```powershell
py -3.11 train.py --algo rcpo --constraint-allocation-drawdown --config configs/rcpo_allocation_drawdown_penalty.yaml
```

Train RCPO with reward correction:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --use-drc --config configs/default.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --use-gdrc --config configs/default.yaml
```

Train with Gaussian reward noise after setting `reward_noise.enabled: true` in the YAML:

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
py -3.11 train.py --algo rcpo --constraint-drawdown --resume-run-dir "runs\simplex_v1_rcpo_none_20260606_150515\seed_0"
```

Resume RCPO with GDRC:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --use-gdrc --resume-run-dir "runs\noise_v1_rcpo_gdrc_20260429_121506\seed_0"
```

Resume PPO:

```powershell
py -3.11 train.py --algo ppo_unconstrained  --resume-run-dir "runs\simplex_v1_ppo_unconstrained_none_20260526_125507\seed_0"
```

To resume from a specific checkpoint file inside the run directory:

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --resume-run-dir "runs\new_rcpo_none_YYYYMMDD_HHMMSS\seed_0" --resume-checkpoint checkpoint_best.pt
```

Resume the soft allocation-constraint RCPO baseline:

```powershell
py -3.11 train.py --algo rcpo --constraint-allocation --resume-run-dir "runs\rcpo_allocation_penalty_none_YYYYMMDD_HHMMSS\seed_0"
```

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

## Action Constraints

By default, `configs/default.yaml` uses `environment.action_mode: simplex_decomposition`.
The policy emits either branch logits or branch simplex weights, then the environment maps them through four simplex-decomposition branches before trading. This makes the final portfolio weights long-only, sum to one, and satisfy two hard allocation constraints:

```yaml
environment:
  action_mode: simplex_decomposition
  simplex_action_format: branch_weights
  initial_portfolio_mode: constrained_neutral
  allocation_constraint_1_indices: [1, 3, 5]
  allocation_constraint_2_indices: [4, 5]
  active_constraint_preset: c3

network:
  policy_architecture: simplex_autoregressive_dirichlet
  branch_credit_mode: standalone
```

The active preset `c3` requires at least `0.55` portfolio weight in each custom allocation set. Presets `c1` and `c2` provide looser alternatives. New episodes begin at the constrained-neutral CAOSD allocation, so the first action pays turnover only for moving away from that feasible baseline, not for an artificial all-cash rebalance.

Available policy architectures:

- `flat_gaussian`: old one-head actor. With simplex decomposition, the env splits its flat logits into CAOSD branches.
- `simplex_branch_gaussian`: shared encoder plus four parallel Gaussian branch heads.
- `simplex_autoregressive_gaussian`: shared encoder plus four autoregressive Gaussian-logit branch heads. Later heads condition on previous branch softmax allocations.
- `simplex_autoregressive_dirichlet`: shared encoder plus four autoregressive Dirichlet heads. Each head samples a valid within-branch simplex allocation directly.

For v3, set:

```yaml
network:
  policy_architecture: simplex_autoregressive_dirichlet
  branch_credit_mode: standalone
  dirichlet_init_concentration: 1.5
  dirichlet_min_concentration: 0.5
  dirichlet_max_concentration: 8.0
```

The config loader resolves Gaussian architectures to `branch_logits` and applies softmax within each branch. Dirichlet resolves to `branch_weights`, because its sampled action already lies on each branch simplex.

With `branch_credit_mode: standalone`, each branch receives its own shadow-portfolio return and drawdown advantage, and its PPO loss is scaled by its realized CAOSD coefficient `z_i`. RCPO still uses one global Lagrange multiplier, updated only from the final combined portfolio drawdown cost. Set `branch_credit_mode: global` to preserve the original joint log-probability and shared final-portfolio advantage.

## Constraint Definition

Reward is net portfolio log return after transaction costs:

```text
reward_t = log(1 + net_simple_return_t)
```

RCPO now supports three active constraint modes. Select exactly one at the command line:

- `--constraint-drawdown`: benchmark-relative maximum drawdown control.
- `--constraint-allocation`: soft allocation-constraint penalty for the non-simplex baseline.
- `--constraint-allocation-drawdown`: one Lagrange cost combining soft allocation violations and drawdown risk.

### Drawdown RCPO

Drawdown RCPO tracks the agent portfolio path and an online benchmark path under the same transaction-cost model. In simplex mode, `configs/default.yaml` uses the constrained-neutral CAOSD baseline as this benchmark; set `drawdown_benchmark_mode: true_equal_weight` to use the old true equal-weight benchmark instead.

```text
agent_current_drawdown_t = (agent_running_peak_t - agent_portfolio_value_t) / agent_running_peak_t
agent_max_drawdown_t = max(previous_agent_max_drawdown, agent_current_drawdown_t)
benchmark_max_drawdown_t = max(previous_benchmark_max_drawdown, benchmark_current_drawdown_t)
budget_t = max(drawdown_budget_floor, benchmark_drawdown_margin * benchmark_max_drawdown_t)
drawdown_violation_t = max(0, agent_max_drawdown_t - budget_t)
constraint_cost = drawdown_violation^2 / drawdown_cost_scale
```

Default settings:

```yaml
environment:
  drawdown_budget_floor: 0.05
  drawdown_benchmark_mode: constrained_neutral
  benchmark_drawdown_margin: 0.90
  drawdown_cost_scale: 0.10

rcpo:
  alpha: null
  alpha_budget_ratio: 0.04
```

This means the policy can seek return, but RCPO penalizes episode paths whose running maximum drawdown exceeds a budget defined online from the selected benchmark. The `0.90` margin asks the policy to target about 10% less drawdown than the benchmark, subject to the `0.05` minimum floor. With `alpha_budget_ratio: 0.04`, the Lagrange multiplier tolerates about 4% of the current effective drawdown budget as average violation before it increases.

### Soft Allocation-Penalty RCPO Baseline

`configs/rcpo_allocation_penalty.yaml` uses `environment.action_mode: softmax`, so the policy is not protected by simplex decomposition. Instead, RCPO uses the two allocation constraint violations as its active cost:

```text
raw_violation = allocation_constraint_1_violation_cost
              + allocation_constraint_2_violation_cost
constraint_cost = raw_violation / allocation_constraint_cost_scale
```

Default healthy settings for this baseline:

```yaml
environment:
  action_mode: softmax
  drawdown_benchmark_mode: constrained_neutral
  constraint_mode: allocation
  allocation_constraint_cost_scale: 20.0

rcpo:
  constraint_mode: allocation
  alpha: 0.0001
  lambda_lr_up: 0.001
  lambda_lr_down: 0.03
```

This baseline answers a clean research question: can a soft Lagrange penalty learn to reduce allocation violations without destroying return, and how does that compare with simplex decomposition, which guarantees allocation feasibility by construction? The policy remains plain softmax, but `constrained_neutral` uses the same CAOSD-neutral feasible allocation for the initial portfolio and drawdown benchmark diagnostics.

### Combined Allocation And Drawdown RCPO

`configs/rcpo_allocation_drawdown_penalty.yaml` keeps the same non-simplex softmax policy but optimizes one combined constraint stream:

```text
constraint_cost = allocation_constraint_cost
                + combined_drawdown_cost_weight * drawdown_constraint_cost
```

The default drawdown weight is `0.25`, with `alpha: 0.00015` and `lambda_lr_up: 0.0005`. These conservative values keep the drawdown term secondary to allocation feasibility and reduce the risk that the Lagrange penalty overwhelms the return advantage early in training.

To compare the soft baseline against simplex runs after training:

```powershell
py -3.11 scripts\compare_allocation_penalty_vs_simplex.py --allocation-run-dir "runs\rcpo_allocation_penalty_none_YYYYMMDD_HHMMSS\seed_0"
```

## DRC / GDRC Reward Correction

Optional reward-correction modes are available for PPO and RCPO:

- No flag: train on the environment reward directly.
- `--use-drc`: train a single distributional reward critic and use corrected rewards for reward advantages/value targets.
- `--use-gdrc`: train a fine-bin GDRC ensemble and select between 48-bin and 64-bin reward critics.

DRC/GDRC only change the reward stream used for training. Evaluation summaries and plots still report actual portfolio returns from the environment.
They currently require `network.branch_credit_mode: global`; branch-aware reward correction is intentionally deferred.

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

`configs/default.yaml` includes an optional Gaussian training reward-noise channel. It is disabled by default for the simplex-decomposition phase.

```yaml
reward_noise:
  enabled: false
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
- `checkpoint_best.pt` for maximum validation excess return
- `checkpoint_best_feasible.pt` for maximum validation excess return with validation constraint cost at or below validation alpha (RCPO only, when found)
- `training_summary.json`
- `evaluation/`
- `evaluation_best/`
- `evaluation_best_feasible/` when a feasible RCPO checkpoint exists
- `evaluation_last/`

Evaluation folders contain JSON summaries and PNG plots for cumulative return, mean cumulative return across future branches, portfolio weights, turnover, drawdown, drawdown constraint cost, and lambda trajectory.

## Tests

```powershell
py -3.11 -m pytest
```

py -3.11 train.py --algo ppo_unconstrained --resume-run-dir "runs\simplex_v2.4_ppo_gaussian_ppo_unconstrained_none_20260718_162406\seed_0"

py -3.11 train.py --algo rcpo --constraint-drawdown --resume-run-dir "runs\simplex_v2.4_rcpo_gaussian_rcpo_none_20260718_162411\seed_0"

py -3.11 train.py --algo ppo_unconstrained --resume-run-dir "runs\simplex_v2.4_ppo_dirichlet_ppo_unconstrained_none_20260718_162418\seed_0"

py -3.11 train.py --algo rcpo --constraint-drawdown --resume-run-dir "runs\simplex_v2.4_rcpo_dirichlet_rcpo_none_20260718_162422\seed_0"

py -3.11 train.py --algo rcpo --constraint-allocation --resume-run-dir "runs\rcpo_allocation_penalty_v2_rcpo_none_20260718_162428\seed_0"

py -3.11 train.py --algo rcpo --constraint-allocation-drawdown --resume-run-dir "runs\rcpo_allocation_drawdown_penalty_rcpo_none_20260718_162434\seed_0"