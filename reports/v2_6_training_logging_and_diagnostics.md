# V2.6 Training Logging And Diagnostics Recommendations

## Goal

Make PPO/RCPO simplex training logs smaller, easier to read, and able to answer the important debugging questions:

1. Is a Dirichlet policy becoming too concentrated or hitting its bounds?
2. Are reward and cost critics predicting their targets well?
3. For each CAOSD branch, is `lambda * A_cost` stronger than `A_reward`?
4. Did PPO stop early because of KL, and which branch caused it?
5. Is a displayed validation result new, or only the last cached validation result?

This is a logging and presentation proposal. It does not change the portfolio reward, constraint definition, PPO objective, or RCPO objective.

## Current Findings

The current V2.5 `metrics.jsonl` files have about 157--161 fields per update and are about 7 KB per row. A 120,000-update run is about 0.84--0.87 GB.

The current code already records:

- global reward/cost advantage standard deviations;
- branch reward/cost advantage standard deviations;
- branch CAOSD masses `z1..z4`;
- joint and per-branch approximate KL;
- per-branch entropy;
- optimizer steps completed and the KL that triggered early stop;
- reward/cost value losses.

The current code does not record:

- actual Dirichlet concentration parameters;
- critic explained variance;
- per-branch RCPO reward-versus-cost advantage ratios;
- the number and magnitude of lambda updates in one rollout.

`batch_concentration_mean` must not be used as a Dirichlet diagnostic. It is final portfolio concentration, `sum(weights ** 2)`, not a Dirichlet concentration parameter.

## New Per-Branch Diagnostics

Only write branch diagnostics for a branch whose `branch_train_mask` is true. Write `null` for an inactive branch rather than `0`; zero can mean a real value while `null` clearly means the branch was not trained.

### 1. Dirichlet Concentration

For `simplex_autoregressive_dirichlet`, collect the concentration vector `alpha_i` generated for each active branch during rollout collection. Aggregate it per rollout rather than storing every step and every component.

Recommended fields:

```text
dirichlet_alpha0_branch_i
dirichlet_alpha_component_mean_branch_i
dirichlet_alpha_component_min_branch_i
dirichlet_alpha_component_max_branch_i
dirichlet_alpha_lower_bound_rate_branch_i
dirichlet_alpha_upper_bound_rate_branch_i
```

where:

```text
alpha0_i = sum_j alpha_i,j
```

Interpretation:

- a high and increasing `alpha0` means samples are increasingly concentrated around the policy mean;
- frequent upper-bound hits indicate the policy is becoming nearly deterministic because of the configured maximum concentration;
- frequent lower-bound hits indicate sparse or extreme Dirichlet samples;
- component min/max are needed because the same `alpha0` can describe either a balanced or an asymmetric branch distribution.

For Gaussian architectures, do not write fake Dirichlet fields. Instead, optionally record `gaussian_policy_std_mean_branch_i` and `gaussian_policy_std_min_branch_i` under Gaussian-specific names.

### 2. Critic Explained Variance

Value loss alone is not a reliable critic-quality diagnostic because reward and cost targets have different scales. Add explained variance calculated on the rollout before optimizer updates:

```text
EV(target, prediction) =
    1 - Var(target - prediction) / (Var(target) + epsilon)
```

Recommended fields:

```text
reward_critic_ev
cost_critic_ev
branch_reward_critic_ev_i
branch_cost_critic_ev_i
```

Use the rollout return targets and the old rollout value predictions. This measures whether the critic that collected the rollout was useful, rather than measuring how well it can fit a batch after several optimizer passes.

Interpretation:

- `EV` near `1`: useful critic;
- `EV` near `0`: no better than predicting the batch mean;
- `EV < 0`: worse than the constant-mean baseline.

For PPO, cost critics are not optimized. Record `cost_critic_ev` and `branch_cost_critic_ev_i` as `null` for PPO so that an untrained cost head is not misinterpreted as a meaningful diagnostic.

### 3. Per-Branch RCPO Cost-To-Reward Strength

The mean of an advantage is normally near zero, so it is not a reliable denominator. Record the relative magnitude before branch advantage normalization:

```text
branch_lambda_cost_adv_ratio_i =
    abs(lambda) * RMS(A_cost_i)
    / max(RMS(A_reward_i), epsilon)
```

`A_cost_i` must be the actual cost-advantage stream used by branch `i`:

- current standalone-credit implementation: branch-local cost advantage;
- later global-cost design: final portfolio cost advantage supplied to the branch;
- later counterfactual design: that branch's counterfactual cost advantage.

Also record:

```text
branch_combined_advantage_std_i
branch_reward_cost_adv_correlation_i
```

The ratio is a diagnostic, not a literal percentage of the policy gradient. PPO normalizes the selected advantage afterward. A ratio substantially above `1` still means cost is likely steering the branch more strongly than reward before normalization.

### 4. Lambda Audit Fields

Record:

```text
lambda_before
lambda_after
lambda_delta
lambda_update_count
lambda_gap
```

This is important because the current RCPO implementation updates lambda once per PPO epoch. With four epochs, the same rollout cost currently causes four multiplier updates. The fields make the effective multiplier schedule auditable and remain useful if lambda is changed to one update per rollout later.

## Keep In Per-Update Metrics

Keep the following compact training fields in `metrics.jsonl`:

- `update`, learning rate, wall-clock update time;
- rollout reward and relative wealth versus constrained-neutral baseline;
- turnover;
- active constraint cost, active budget, drawdown gap, and alpha target;
- lambda audit fields and global cost-to-reward advantage ratio for RCPO;
- policy loss, reward/cost value loss, critic explained variance;
- joint KL, clip fraction, optimizer steps completed, and early-stop trigger KL;
- branch `z_i`, branch KL, branch entropy, active-branch reward/cost ratios, and branch critic EV;
- Dirichlet concentration aggregates only for Dirichlet runs.

For a hard-simplex run, allocation violations can be checked through a compact `allocation_feasible` flag and a maximum violation value. They do not need a long collection of near-zero metrics on every update unless a violation appears.

## Move Validation To A Separate Log

The current training row repeats the full last validation summary even when `validation_evaluated=0`. This has two problems:

1. it makes every `metrics.jsonl` row much larger;
2. terminal output can look like a new validation result when it is only stale cached data.

Recommended layout:

```text
metrics.jsonl             # compact per-update training metrics
validation_metrics.jsonl  # one complete row only when validation runs
run_metadata.json         # static run information, optional convenience copy
config_snapshot.yaml      # authoritative configuration
```

`metrics.jsonl` should retain only `validation_evaluated`, the validation score when it is newly computed, and checkpoint flags. Full return, drawdown, allocation, and branch validation summaries should go to `validation_metrics.jsonl`.

## Do Not Save Or Print When Inactive

### Static Configuration

Do not repeat these on every update; they already belong in `config_snapshot.yaml` and checkpoint metadata:

- algorithm, device, action mode, policy architecture, simplex action format;
- constraint preset and static allocation indices;
- reward-correction mode and reward-noise configuration;
- validation interval, turnover cap, alpha mode, and other static hyperparameters.

### Reward Noise And Reward Correction

When `reward_noise.enabled=false`, omit per-update noise mean/std fields.

When `reward_correction.mode=none`, omit all of:

```text
observed_reward_mean
corrected_reward_mean
reward_correction_delta_*
reward_correction_oce
reward_correction_clamp_rate
gdrc_selected_bins
gdrc_candidate_bins
gdrc_reward_min
gdrc_reward_max
```

Do not print placeholders such as `reward_delta_abs=0`, `reward_oce=0`, or `gdrc_bins=0` in ordinary clean-reward training.

If DRC/GDRC is actively studied, write its detailed diagnostics to a dedicated `reward_correction_metrics.jsonl` instead of bloating every generic training row.

### Other Conditional Fields

- Do not save diversification-cost diagnostics when diversification is disabled.
- Do not save allocation-plus-drawdown combined-cost diagnostics unless that combined mode is active.
- Do not print inactive branch fields.
- For hard simplex constraints, hide near-zero allocation violation details from the normal terminal line; print a warning only if tolerance is exceeded.

## Terminal Output Proposal

Print a compact training line every update or at a chosen interval:

```text
u=120000 lr=1e-4 ret=... vs_base=... turn=...
KL=0.0023 steps=16/16 clip=... lambda=0.582 cost/alpha=...
z=(0.00,0.50,0.26,0.24) global_risk_ratio=0.65
```

For RCPO, print branch diagnostics periodically, for example every 20--50 updates:

```text
branch_ratio=(inactive,2.33,5.68,0.32)
branch_EV_R=(--,0.31,0.18,0.42)
branch_EV_C=(--,0.09,-0.14,0.27)
```

For Dirichlet runs, add:

```text
alpha0=(--,4.8,6.1,5.4) bound_hit=(--,0%,3%,0%)
entropy=(--,...)
```

Only print a full validation line when validation was actually run:

```text
VAL: +x.xx% vs constrained-neutral | cost=... alpha=...
MDD=... benchmark=... budget=... feasible=yes/no
```

When `validation_evaluated=0`, do not print the prior validation metrics again.

## Implementation Locations

- `src/rcpo_portfolio/models.py`: expose branch Dirichlet concentrations in `PolicyOutput`.
- `src/rcpo_portfolio/rollouts.py`: store rollout-level concentration aggregates and calculate critic explained variance from rollout targets and pre-update predictions.
- `src/rcpo_portfolio/algorithms/ppo.py`: retain existing KL/entropy metrics and add branch combined-advantage diagnostics if not calculated in rollout collection.
- `src/rcpo_portfolio/trainer.py`: calculate branch lambda ratios, record lambda audit fields, split validation logging, and conditionally format terminal output.
- `src/rcpo_portfolio/reward_correction/base.py`: return a minimal empty metric dictionary for `NoRewardCorrector`, rather than a large block of zero-valued GDRC placeholders.

## Acceptance Checks

1. A clean Gaussian PPO run has no reward-correction/noise metrics and no Dirichlet fields.
2. A clean Dirichlet RCPO run has active-branch alpha diagnostics, branch reward/cost EV, and branch cost-to-reward ratios.
3. Inactive branch fields are `null`, not numeric zero placeholders.
4. PPO cost critic EV is `null`.
5. Full validation metrics appear only in `validation_metrics.jsonl` when validation runs.
6. Terminal output never presents stale validation results as fresh results.
7. Existing resume and plot loaders remain backward compatible with old `metrics.jsonl` files.

