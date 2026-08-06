# V2.6 Recommendation: Benchmark-Relative Drawdown State And Counterfactual Credit

## Purpose

This document records two linked improvements for the next simplex + RCPO experiment. They align the information and branch-credit signal with a benchmark-relative current-drawdown constraint.

The existing V2.5 runs remain unchanged and serve as the comparison baseline.

## Active RCPO Constraint

The active RCPO cost is benchmark-relative current drawdown, not repeated maximum-drawdown occupancy cost:

```text
benchmark_drawdown_margin = 0.95

effective_drawdown_budget_t =
    max(drawdown_budget_floor,
        0.95 * benchmark_current_drawdown_t)

drawdown_gap_t = agent_current_drawdown_t - effective_drawdown_budget_t

constraint_cost_t =
    max(0, drawdown_gap_t)^2 / drawdown_cost_scale
```

The policy is charged on every day that it remains more underwater than the constrained-neutral benchmark-relative budget. When the agent recovers below the budget, the active cost returns to zero. Maximum drawdown remains an observation feature and final evaluation metric.

## 1. Add Drawdown State To The Observation

### Problem

Current drawdown depends on portfolio history, but the current observation does not expose the relevant history. The policy and cost critic see trailing market returns, weights, turnover, and allocation diagnostics, but they do not directly know whether the portfolio is currently more underwater than the benchmark-relative budget.

This makes the drawdown-constrained decision problem partially observable. Two identical-looking market observations can require different actions if one portfolio is currently close to the drawdown budget and the other is not.

### New Observation Features

Add the following normalized scalar features to `PortfolioEnv._get_observation()`:

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
agent_current_drawdown_t =
    (agent_running_peak_t - agent_portfolio_value_t) / agent_running_peak_t

agent_max_drawdown_t =
    max(agent_max_drawdown_(t-1), agent_current_drawdown_t)

effective_drawdown_budget_t =
    max(drawdown_budget_floor,
        0.95 * benchmark_current_drawdown_t)

drawdown_gap_t =
    agent_current_drawdown_t - effective_drawdown_budget_t

episode_progress_t = steps_elapsed / episode_length
```

`benchmark_current_drawdown` and `benchmark_max_drawdown` should be retained even though the budget is already present. The budget can be limited by the drawdown floor, so it does not always reveal the benchmark's full state.

### Expected Effect

- The policy can become defensive before it remains more underwater than the benchmark-relative budget.
- The global cost critic receives the state variables that determine the active constraint cost.
- The episode becomes substantially closer to a Markov decision process for the relative underwater-risk objective.

## 2. Use Counterfactual Difference Credit For Both Return And Cost

### Problem

The current standalone branch signal treats each CAOSD branch as a fully invested shadow portfolio. This is not the same as the branch's actual contribution to the final portfolio:

```text
final_weights_t = z1 * y1_t + z2 * y2_t + z3 * y3_t + z4 * y4_t
```

In particular, a branch with small `z_i` can receive a large standalone drawdown cost even though it has limited final portfolio exposure. Branch 2 can also change `z3` and `z4`, which is not represented by its standalone return or cost.

### Counterfactual Construction

For the realized joint branch action:

```text
a_t = (a1_t, a2_t, a3_t, a4_t)
w_actual_t = CAOSD(a_t)
```

For every active branch `i`, create a counterfactual action that replaces only that branch with its neutral action while holding all other realized branch actions fixed:

```text
a_cf_i,t = (a1_t, ..., neutral_action_i, ..., a4_t)
w_cf_i,t = CAOSD(a_cf_i,t)
```

The entire CAOSD mapper must be rerun for each counterfactual. Do not replace only a padded branch weight in the final portfolio, because changing a branch can alter `z1`, `z2`, `z3`, and `z4`.

Neutral action by policy type:

```text
Gaussian branch logits: zeros for the selected branch segment
Dirichlet branch weights: uniform weights within the selected branch
Inactive/fixed branch: no counterfactual path is required
```

### Counterfactual Reward And Cost

Maintain one actual portfolio path and one counterfactual shadow path per active branch. Each shadow path must keep its own previous weights, portfolio value, running peak, current drawdown, and maximum drawdown. All paths use the same realized market return at each step.

```text
r_actual_t = log(1 + net_return_actual_t)
r_cf_i,t = log(1 + net_return_cf_i,t)

delta_reward_i,t = r_actual_t - r_cf_i,t
```

The actual and counterfactual benchmark-relative current-drawdown costs are:

```text
c_actual_t =
    max(0, agent_current_drawdown_t - effective_drawdown_budget_t)^2
    / drawdown_cost_scale

c_cf_i,t =
    max(0, counterfactual_current_drawdown_i,t - effective_drawdown_budget_t)^2
    / drawdown_cost_scale

delta_cost_i,t = c_actual_t - c_cf_i,t
```

Interpretation:

```text
delta_reward_i > 0: branch i improves final portfolio return versus neutral
delta_reward_i < 0: branch i reduces final portfolio return versus neutral

delta_cost_i > 0: branch i leaves the final portfolio more underwater than neutral
delta_cost_i < 0: branch i helps the final portfolio recover relative to neutral
```

Example:

```text
benchmark current drawdown = 10%
effective budget = 9.5%  # 0.95 * benchmark current drawdown
actual current drawdown = 13%
counterfactual current drawdown after neutralizing branch 2 = 11%
drawdown_cost_scale = 0.10

c_actual = (0.13 - 0.095)^2 / 0.10 = 0.01225
c_cf_2  = (0.11 - 0.095)^2 / 0.10 = 0.00225

delta_cost_2 = 0.01000
```

Branch 2 receives a positive risk contribution and should be discouraged from repeating the corresponding action.

### PPO And RCPO Training Target

Use a separate return critic and risk-difference critic for each active branch:

```text
A_reward_i = GAE(delta_reward_i, V_reward_i)
A_cost_i = GAE(delta_cost_i, V_cost_i)
```

For PPO:

```text
A_i = A_reward_i
```

For RCPO:

```text
A_i = A_reward_i - lambda * A_cost_i
```

The branch actor objective is:

```text
L_policy =
    -mean_t sum_i min(
        ratio_i,t * A_i,t,
        clip(ratio_i,t, 1 - epsilon, 1 + epsilon) * A_i,t
    )
```

Do not multiply this objective by `z_i` again. The counterfactual difference already measures the branch's realized contribution through the full CAOSD mapping, including the branch mass and dynamic `z` coefficients. Multiplying by `z_i` would double-downweight low-mass branches.

### Global Lambda Update Remains Unchanged In Principle

Keep one global lambda. It is updated from the actual final portfolio constraint cost, never from branch difference costs:

```text
lambda <- max(
    0,
    lambda + lambda_lr * (mean(c_actual) - alpha)
)
```

`delta_cost_i` can be negative and is only a branch-credit signal. It is not a portfolio constraint measurement.

For V2.6, update lambda once per rollout rather than once per PPO epoch. This avoids repeatedly applying the same rollout cost gap merely because the optimizer uses multiple epochs.

### Computational Cost

No second market environment, second random-number stream, or additional market rollout is needed. The implementation uses the same market return for the actual path and the branch counterfactual paths.

At each environment step, compute:

```text
1 actual CAOSD mapping and portfolio transition
up to 4 counterfactual CAOSD mappings and shadow portfolio transitions
```

This adds small vector operations relative to actor forward passes and PPO backpropagation.

## Implementation Checklist

- Extend the observation with all seven drawdown/benchmark/progress features.
- Extend episode state with counterfactual weights, values, peaks, current drawdowns, and maximum drawdowns for four branches.
- Reset every counterfactual shadow path from the same constrained-neutral initial portfolio.
- Recompute the full CAOSD mapping after neutralizing each active branch.
- Include counterfactual turnover using each shadow path's own previous final weights.
- Store `branch_delta_rewards` and `branch_delta_costs` in rollout storage.
- Train branch return and cost critics on GAE from these difference streams.
- Keep the global lambda update based only on actual final portfolio cost.
- Log per-branch mean difference reward, mean difference cost, critic loss, explained variance, and advantage scale.
- Preserve maximum drawdown as an evaluation metric while reporting the active benchmark-relative current-drawdown cost separately.

## Acceptance Criteria For A Pilot Run

- All simplex policies continue to have zero allocation-constraint violations by construction.
- The drawdown state is visible in every policy and critic observation.
- A branch with `z_i = 0` has an approximately zero counterfactual difference and no meaningful actor gradient.
- Counterfactual paths use no future market information.
- RCPO lambda changes only from actual final portfolio constraint cost.
- Branch risk signals can be positive or negative, while actual portfolio constraint cost remains nonnegative.
- Metrics make it possible to compare branch return contribution, branch relative-underwater contribution, and final portfolio maximum drawdown.
