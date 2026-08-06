# V2.6 Recommendation: Lambda Cadence, KL Safety, And Checkpoint Selection

## Purpose

This document records three training-loop changes for the next RCPO/Simplex experiment. These changes do not alter the market, allocation constraints, CAOSD feasibility mapping, or reward definition.
The active RCPO cost is benchmark-relative current drawdown:

```text
effective_budget_t =
    max(drawdown_budget_floor,
        0.95 * constrained_neutral_current_drawdown_t)

constraint_cost_t =
    max(0, agent_current_drawdown_t - effective_budget_t)^2
    / drawdown_cost_scale
```

Use `benchmark_drawdown_margin: 0.95`. Maximum drawdown remains a reporting and evaluation metric, not the repeated per-step active cost.


## 1. Update Lambda Once Per Rollout

### Current Behavior

One rollout supplies one fixed batch constraint measurement:

```text
gap = batch_constraint_cost_mean - alpha
```

The current RCPO update repeats the Lagrange update once for every PPO optimizer epoch, even though no new portfolio path or new constraint observation is collected:

```text
for epoch in range(optimization.epochs):
    lambda = max(0, lambda + lambda_lr * gap)
```

Therefore, the effective lambda rate depends on the number of PPO epochs. For example, with `lambda_lr_up: 0.00075`:

```text
3 epochs -> effective increase = 0.00225 * gap per rollout
4 epochs -> effective increase = 0.00300 * gap per rollout
```

This is especially undesirable when PPO stops early because of KL: lambda can still be updated for the configured epoch count despite fewer completed actor updates.

### Required Change

Update lambda exactly once after the PPO/RCPO optimization pass for that rollout:

```text
gap = batch_constraint_cost_mean - alpha
lr = lambda_lr_up if gap > 0 else lambda_lr_down

lambda = max(0, lambda + lr * gap)
```

The updated lambda is used for the next rollout. The current rollout's actor optimization continues to use the lambda value that existed when its advantages were formed.

### Expected Effect

- One rollout gives one piece of new constraint evidence and one dual update.
- Changing PPO epochs affects policy optimization only, not the effective constraint learning rate.
- Gaussian and Dirichlet runs become directly comparable when they use different epoch counts.
- Lambda should grow more smoothly and be less likely to dominate reward advantages.

## 2. Use A Non-Negative KL Estimator And Stop Before A Violating Update

### Current Behavior

The current approximate KL estimate is:

```text
approx_kl = mean(old_log_prob - new_log_prob)
```

For a finite sampled minibatch this estimate can be negative, particularly for the Dirichlet policy. A negative estimate is not interpretable as a KL divergence and can allow a large policy change to avoid the KL early-stop condition.

The current loop also checks KL after `optimizer.step()`. Thus, a minibatch that already exceeds `target_kl` receives one extra update before training stops.

### Required Change

For each minibatch, define:

```text
log_ratio = new_log_prob - old_log_prob
ratio = exp(log_ratio)

approx_kl = mean(ratio - 1 - log_ratio)
```

This estimator is non-negative up to numerical precision and is a more stable estimate of policy divergence.

Compute this value before backward/optimizer update. If it exceeds the configured KL target, stop the remaining minibatches and epochs without applying the violating minibatch update:

```text
if approx_kl > target_kl:
    trigger_minibatch_kl = approx_kl
    stop optimization loop
else:
    backward()
    optimizer.step()
```

Apply the same estimator branch by branch for `approx_kl_branch_1` through `approx_kl_branch_4` diagnostics. The joint KL remains the early-stop criterion.

### Expected Effect

- KL logs become meaningful and non-negative.
- Dirichlet runs can no longer appear artificially safe merely because a noisy KL estimate is negative.
- PPO trust-region protection becomes stricter: no known violating minibatch is applied.
- `optimizer_steps_completed` and `trigger_minibatch_kl` remain useful diagnostics.

## 3. Save Two Model-Selection Checkpoints

Save these selected models for every PPO and RCPO run:

### `checkpoint_best_return.pt`

Selection rule:

```text
maximum validation_mean_excess_cumulative_return
```

This measures the maximum validation return found by the policy and may violate
the configured constraint. Do not create the redundant `checkpoint_best.pt` alias.

### `checkpoint_best_feasible.pt`

Selection rule:

```text
primary key: maximum validation feasible_branch_rate
tie-breaker: maximum validation_mean_excess_cumulative_return
```

This checkpoint prioritizes broad constraint satisfaction across validation
branches. A checkpoint with a higher feasible rate wins regardless of return;
return is compared only when feasible rates are equal. The file is created after
the first validation, so it can exist before an 80% or 100% diagnostic tier is
reached.

### `checkpoint_last.pt` Recovery Snapshot

Overwrite this file after every completed update only for interruption recovery
and training-stability diagnosis. It is not a selected model and is not the
default model for final reported performance.
## Final Evaluation Protocol

Checkpoint selection must use validation branches only. Test/future branches must never influence the checkpoint choice.

For RCPO, report both:

```text
Best Return: checkpoint_best_return.pt
Best Feasible Rate: checkpoint_best_feasible.pt
```

Evaluate `checkpoint_last.pt` only as a training-stability diagnostic.

## Acceptance Criteria

- Lambda changes once per rollout, regardless of PPO epoch count or KL early stop.
- KL diagnostics are non-negative up to small floating-point tolerance.
- A KL-violating minibatch is not passed to `optimizer.step()`.
- `checkpoint_best_return.pt` always reflects the maximum validation return score.
- `checkpoint_best_feasible.pt` maximizes feasible-branch rate first and validation return second.
- `checkpoint_last.pt` remains resumable and is written after each completed update.
- Evaluation summaries clearly identify which checkpoint was evaluated.
