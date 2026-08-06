# V2.6 Phase 1 Reference: Relative Current Drawdown And Training-Core Corrections

## Purpose

Phase 1 creates a clean successor to V2.5 without introducing counterfactual branch credit yet. It changes the active RCPO constraint from repeated maximum-drawdown occupancy to benchmark-relative current drawdown, exposes the state required to learn that constraint, aligns branch RCPO cost credit with the actual final portfolio, and corrects lambda cadence, KL safety, checkpoint selection, and logging.

The V2.5 runs and artifacts remain unchanged and serve as the experimental baseline. Phase 1 must start new V2.6 runs because the observation dimension, constraint semantics, and RCPO credit stream change.

## Scope And Non-Goals

Phase 1 includes:

- benchmark-relative current-drawdown constraint;
- drawdown state in the observation;
- standalone branch return credit with global final-portfolio cost credit;
- one lambda update per rollout;
- non-negative KL estimation and pre-update rejection;
- strict and robust validation-feasible checkpoint selection;
- compact, auditable training and validation logs.

Phase 1 does not include:

- counterfactual reward or cost paths;
- minimum-constraint or Pareto checkpoints;
- reward noise or DRC/GDRC experiments;
- changes to CAOSD hard allocation feasibility;
- claims that the active cost directly constrains episode maximum drawdown.

Maximum drawdown remains an important evaluation metric. The active Phase 1 constraint measures relative current underwater risk and should be named accordingly.

## 1. Active Constraint Semantics

Use a new explicit constraint mode:

```yaml
environment:
  constraint_mode: relative_current_drawdown
  drawdown_benchmark_mode: constrained_neutral
  benchmark_drawdown_margin: 0.90
  drawdown_budget_floor: 0.05
  drawdown_cost_scale: 0.10

rcpo:
  constraint_mode: relative_current_drawdown
  alpha: null
  alpha_budget_ratio: 0.05
```

Do not save this mode as `max_drawdown`. The cost is based on current drawdown and can return to zero after recovery.

At step `t`, update the agent and constrained-neutral benchmark using the same realized market return:

```text
agent_current_drawdown_t =
    (agent_running_peak_t - agent_portfolio_value_t)
    / agent_running_peak_t

benchmark_current_drawdown_t =
    (benchmark_running_peak_t - benchmark_portfolio_value_t)
    / benchmark_running_peak_t
```

The online budget is:

```text
effective_drawdown_budget_t =
    max(
        drawdown_budget_floor,
        benchmark_drawdown_margin * benchmark_current_drawdown_t
    )
```

With the Phase 1 defaults:

```text
effective_drawdown_budget_t =
    max(0.05, 0.90 * benchmark_current_drawdown_t)
```

The active violation and cost are:

```text
drawdown_gap_t =
    agent_current_drawdown_t - effective_drawdown_budget_t

drawdown_violation_t = max(0, drawdown_gap_t)

constraint_cost_t =
    drawdown_violation_t^2 / drawdown_cost_scale
```

This is an occupancy-style relative-underwater cost: the policy pays while it remains more underwater than the allowed budget and stops paying after recovery.

Continue tracking and reporting:

```text
agent_max_drawdown
benchmark_max_drawdown
```

They are evaluation diagnostics, not the active Phase 1 cost.

## 2. Dynamic Alpha And Lambda Measurement

Dynamic alpha must use the same budget and cost units as the active constraint.

Define the tolerated current-drawdown violation at each step as a fraction of the current budget:

```text
tolerated_violation_t =
    alpha_budget_ratio * effective_drawdown_budget_t
```

Convert it to constraint-cost units:

```text
alpha_t =
    tolerated_violation_t^2 / drawdown_cost_scale

        = ((alpha_budget_ratio * effective_drawdown_budget_t)^2)
          / drawdown_cost_scale
```

For one rollout:

```text
observed_cost = mean_t(constraint_cost_t)
rollout_alpha = mean_t(alpha_t)
lambda_gap = observed_cost - rollout_alpha
```

Update lambda exactly once after the policy/value optimization pass:

```text
selected_lr = lambda_lr_up if lambda_gap > 0 else lambda_lr_down

lambda_next = max(
    0,
    lambda_before + selected_lr * lambda_gap
)
```

Recommended Phase 1 defaults:

```yaml
rcpo:
  initial_lambda: 0.0
  lambda_lr_up: 0.00075
  lambda_lr_down: 0.01
  alpha: null
  alpha_budget_ratio: 0.05
```

The current rollout actor update uses `lambda_before`. `lambda_next` is used for the next rollout. Changing PPO epochs must not change the number of lambda updates.

The dynamic-alpha interpretation is approximate because feasibility is assessed using averages of squared violations. It means the rollout's average active cost must remain within the average cost implied by a 5% budget-relative violation tolerance; it does not guarantee that every step is within the tolerance.

## 3. Drawdown Observation State

The Phase 1 constraint is history dependent. Add the state variables that determine it to `PortfolioEnv._get_observation()`:

```text
agent_current_drawdown
agent_max_drawdown
benchmark_current_drawdown
benchmark_max_drawdown
effective_drawdown_budget
drawdown_gap
episode_progress
```

Definitions:

```text
episode_progress = steps_elapsed / episode_length
drawdown_gap = agent_current_drawdown - effective_drawdown_budget
```

All drawdown values are fractions and already have a natural scale near `[0, 1]`. Clip defensive numerical outliers to a documented range before insertion, for example:

```text
drawdown features: [0, 1]
drawdown_gap: [-1, 1]
episode_progress: [0, 1]
```

The observation features must be computed from information available up to the current observation. At reset, all agent and benchmark drawdowns are zero, the budget equals the floor, and episode progress is zero.

Add checkpoint metadata:

```text
observation_schema_version: 2
constraint_semantics_version: relative_current_drawdown_v1
```

Old checkpoints with the previous observation dimension or `max_drawdown` semantics must fail resume/evaluation with a clear compatibility error.

## 4. Phase 1 Branch Credit

Keep the existing standalone branch return paths for PPO return credit. Do not use the fully invested standalone branch drawdown as RCPO cost credit.

The active constraint belongs to the actual final CAOSD portfolio, and global lambda is updated from that same actual portfolio cost. Therefore Phase 1 uses:

```text
PPO branch advantage_i = standalone_branch_reward_advantage_i

RCPO branch advantage_i =
    standalone_branch_reward_advantage_i
    - lambda * global_final_portfolio_cost_advantage
```

The global cost advantage is shared across active branches. The existing detached CAOSD mass remains in the branch policy objective because standalone return treats each branch as fully invested:

```text
branch_loss_i = -mean(
    z_i * min(
        ratio_i * normalized_combined_advantage_i,
        clipped_ratio_i * normalized_combined_advantage_i
    )
)
```

Inactive branches remain excluded by `branch_train_mask`. In the current Experiment 2 constraint design, `z1 = 0`, so branch 1 remains inactive.

Use an explicit configuration value so the credit semantics are auditable:

```yaml
network:
  branch_credit_mode: standalone_reward_global_cost
```

The branch cost critics and fully invested branch drawdown costs are not trained or used in this mode. PPO cost diagnostics remain `null` because PPO does not optimize a cost critic.

## 5. PPO/RCPO KL Safety

Replace the signed sampled estimator with a non-negative estimator.

For the joint policy:

```text
log_ratio = clamp(new_log_prob - old_log_prob, -20, 20)
ratio = exp(log_ratio)

approx_kl = mean(ratio - 1 - log_ratio)
```

For each branch:

```text
branch_log_ratio_i =
    clamp(new_branch_log_prob_i - old_branch_log_prob_i, -20, 20)

approx_kl_branch_i =
    mean(exp(branch_log_ratio_i) - 1 - branch_log_ratio_i)
```

The joint KL remains the early-stop criterion. Evaluate KL before backward and `optimizer.step()`:

```text
optimizer_steps_attempted += 1

if approx_kl > target_kl:
    rejected_minibatch_kl = approx_kl
    stop remaining minibatches and epochs
else:
    backward()
    optimizer.step()
    optimizer_steps_completed += 1
```

A rejected minibatch must not alter model or optimizer state.

Recommended optimization starting points:

```yaml
common:
  rollout_steps: 2048
  minibatch_size: 512
  clip_epsilon: 0.10
  target_kl: 0.02

ppo_gaussian:
  epochs: 4
  learning_rate: 0.000085
  learning_rate_final: 0.000040
  entropy_coef: 0.0015

ppo_dirichlet:
  epochs: 4
  learning_rate: 0.00010
  learning_rate_final: 0.000040
  entropy_coef: 0.0020

rcpo_gaussian:
  epochs: 3
  learning_rate: 0.000060
  learning_rate_final: 0.000030
  entropy_coef: 0.0015

rcpo_dirichlet:
  epochs: 4
  learning_rate: 0.000075
  learning_rate_final: 0.000030
  entropy_coef: 0.0020
```

Dirichlet starting bounds:

```yaml
network:
  dirichlet_min_concentration: 0.5
  dirichlet_init_concentration: 1.5
  dirichlet_max_concentration: 8.0
```

## 6. Checkpoint Selection

Save only two model-selection checkpoints:

```text
checkpoint_best_return.pt
checkpoint_best_feasible.pt
```

Do not create `checkpoint_best.pt`. Keep `checkpoint_last.pt` only as a resumable
recovery snapshot; it is not a model-selection checkpoint.

### Best Return

Select by the existing validation score:

```text
maximum validation_mean_excess_cumulative_return
```

It may violate the configured constraint.

### Best Feasible-Rate Checkpoint

Evaluate feasibility separately for every validation branch:

```text
branch_feasible_j =
    branch_average_constraint_cost_j
    <= branch_average_alpha_target_j

feasible_branch_rate = mean_j(branch_feasible_j)
```

Rank every validation checkpoint lexicographically:

```text
1. maximum feasible_branch_rate
2. maximum validation_mean_excess_cumulative_return among equal rates
```

A higher feasible rate always replaces a lower-rate checkpoint, even when its
validation return is lower. At the same feasible rate, only a higher validation
return replaces the checkpoint. Therefore `checkpoint_best_feasible.pt` is created
after the first validation and may initially record a rate below 80%; the
`strict` and `robust_80` tiers remain diagnostic labels only.

Store metadata:

```text
checkpoint_selection: maximum_feasible_branch_rate_then_validation_return
feasibility_tier: strict | robust_80 | null
feasible_branch_rate
validation_score
validation_average_constraint_cost
validation_average_alpha_target
validation_worst_branch_cost_gap
```

Do not create `checkpoint_min_constraint.pt` or `checkpoint_pareto.pt` in Phase 1.

### Last Checkpoint

`checkpoint_last.pt` remains the resumable recovery checkpoint and is not one of
the two selected models or the default reported model.

Test branches must never influence checkpoint selection.
## 7. Logging And Diagnostics

Use three versioned files:

```text
metrics.jsonl             # one compact row per training update
validation_metrics.jsonl  # one complete row only when validation runs
config_snapshot.yaml      # authoritative static configuration
```

Add:

```text
metrics_schema_version: 2
```

Every validation row contains its `update` number. Resume and plotting loaders join training and validation data by update and remain backward compatible with historical runs where validation fields were embedded in `metrics.jsonl`.

### Lambda Audit

Log for RCPO:

```text
lambda_before
lambda_after
lambda_delta
lambda_gap
lambda_update_count       # must equal 1
rollout_constraint_cost_mean
rollout_alpha_mean
```

### KL And PPO Audit

Log:

```text
approx_kl
approx_kl_branch_i
clip_fraction
clip_fraction_branch_i
optimizer_steps_attempted
optimizer_steps_completed
rejected_minibatch_kl
ppo_kl_early_stop
actor_gradient_norm
actor_gradient_norm_branch_i
backbone_gradient_norm
```

Use `null`, not zero, for inactive branches.

### Critic Quality

Calculate explained variance from rollout return targets and pre-update value predictions:

```text
EV(target, prediction) =
    1 - Var(target - prediction) / (Var(target) + epsilon)
```

Log:

```text
reward_critic_ev
cost_critic_ev
branch_reward_critic_ev_i
```

In `standalone_reward_global_cost` mode, branch cost critic EV fields are `null`. PPO global and branch cost EV fields are also `null`.

When target variance is below a documented epsilon, record EV as `null` and also log the target variance.

### Branch Reward/Cost Strength

Before selected-advantage normalization, log:

```text
branch_lambda_cost_adv_ratio_i =
    abs(lambda) * RMS(global_cost_advantage)
    / max(RMS(branch_reward_advantage_i), epsilon)

branch_combined_advantage_std_i
branch_reward_cost_adv_correlation_i
```

These are diagnostics, not literal percentages of policy gradient contribution.

### Dirichlet Diagnostics

For active Dirichlet branches, aggregate rollout concentrations without storing every component at every step:

```text
dirichlet_alpha0_mean_branch_i
dirichlet_alpha0_min_branch_i
dirichlet_alpha0_max_branch_i
dirichlet_alpha_component_mean_branch_i
dirichlet_alpha_component_min_branch_i
dirichlet_alpha_component_max_branch_i
dirichlet_alpha_lower_near_bound_rate_branch_i
dirichlet_alpha_upper_near_bound_rate_branch_i
```

Sigmoid-bounded concentrations never exactly equal their configured bounds. Define near-bound as being within 1% of the configured concentration range from the corresponding bound.

For Gaussian branches, use architecture-specific fields:

```text
gaussian_policy_std_mean_branch_i
gaussian_policy_std_min_branch_i
gaussian_policy_std_max_branch_i
```

### Conditional Logging

Do not repeat static configuration in every metric row. Do not write inactive reward-noise, reward-correction, diversification, combined-constraint, or Dirichlet fields.

For hard simplex policies, write compact allocation diagnostics:

```text
allocation_feasible
allocation_max_violation
```

Emit a warning if hard allocation violation exceeds numerical tolerance.

### Terminal Output

Add a configurable print interval, for example:

```yaml
logging:
  print_interval_updates: 10
  branch_diagnostic_interval_updates: 50
```

Print full validation output only when validation was evaluated on that update. Never print a cached validation result as if it were new.

## 8. Compatibility And Failure Recovery

- New observation dimensions reject V2.5 checkpoint resume.
- New constraint and branch-credit semantics are saved in checkpoint metadata.
- Historical metrics loaders support both embedded validation rows and the new split logs.
- JSONL readers skip or repair a malformed final partial line after interruption.
- Checkpoint files continue to use atomic safe-save behavior.
- Resume reconstructs best-return, strict-feasible, and robust-fallback state from validation logs before continuing.

## 9. Tests

### Environment

- Reset produces zero drawdowns, floor budget, zero gap below budget, and zero progress.
- Positive and negative return sequences produce hand-checked agent and benchmark current drawdowns.
- Recovery makes current-drawdown cost return to zero while maximum drawdown remains recorded.
- Budget equals `max(0.05, 0.90 * benchmark_current_drawdown)` with no future information.
- Observation contains the seven new features with documented bounds.
- Hard simplex allocation violations remain zero.

### Alpha And Lambda

- Per-step alpha matches the budget-ratio formula.
- Rollout alpha is the mean of per-step alpha values.
- Lambda updates once per rollout and uses the pre-update lambda for current actor optimization.
- Lambda cadence is unchanged by PPO epochs or KL rejection.
- Up/down rates and zero floor are correct.

### Branch Credit

- PPO uses standalone branch reward advantages only.
- RCPO uses standalone branch reward plus global actual-portfolio cost advantage.
- No branch cost critic is trained in this mode.
- Inactive branches receive no actor gradient.

### KL

- Joint and branch KL estimates are non-negative up to numerical tolerance.
- A violating minibatch does not change model or optimizer state.
- Attempted, completed, and rejected-step metrics are correct.

### Checkpoints

- Best return is independent of feasibility.
- A higher feasible-branch rate always outranks a lower rate for the feasible checkpoint.
- Equal feasible rates are resolved by the validation return score.
- The feasible-rate checkpoint is created after the first validation, even below 80%.
- Test branches never select checkpoints.

### Logging

- Validation rows are written only when validation runs.
- Resume joins split logs correctly and reads old combined logs.
- Inactive diagnostics use `null` or are omitted according to schema.
- Dirichlet near-bound rates use the 1% tolerance.

## 10. Pilot Acceptance Criteria

Run short, fresh PPO/RCPO Gaussian and Dirichlet pilots before full training. Continue only if:

- hard allocation violations remain zero;
- Gaussian KL early-stop rate is approximately 1% to 15%;
- Dirichlet KL is non-negative and its policy is not persistently at a concentration bound;
- lambda updates exactly once per rollout;
- median branch cost-to-reward advantage ratio is approximately 0.2 to 1.0 and the 95th percentile is below 2.0;
- RCPO reaches at least 80% branch feasibility on validation at some checkpoint;
- relative-current-drawdown cost improves without destroying future relative wealth;
- critic explained variance is not persistently negative.

Use at least three training seeds after the pilot passes. Report best-return and best-feasible checkpoints separately.
