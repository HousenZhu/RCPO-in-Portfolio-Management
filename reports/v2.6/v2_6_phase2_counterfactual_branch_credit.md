# V2.6 Phase 2 Reference: Counterfactual Branch Reward And Cost Credit

## Purpose

Phase 2 investigates whether counterfactual difference credit can improve learning speed and branch-level credit assignment after the Phase 1 relative-current-drawdown training core is stable.

This phase is intentionally separate because counterfactual branch credit changes the actor objective and introduces additional stateful shadow paths. It must not be combined with the initial Phase 1 debugging run. Phase 1 remains the required control experiment.

## Prerequisites

Do not begin Phase 2 until Phase 1 demonstrates:

- correct relative-current-drawdown observations and cost;
- one lambda update per rollout;
- non-negative KL and pre-update rejection;
- reliable split training/validation logs;
- hard allocation feasibility;
- auditable critic quality and branch advantage ratios;
- at least one stable PPO policy and one RCPO run that reaches the robust-feasible validation threshold.

All Phase 2 runs keep the Phase 1 market, allocation constraints, benchmark, dynamic alpha, lambda cadence, KL estimator, checkpoint rules, and evaluation protocol unchanged.

## 1. Existing Structure And Motivation

The current CAOSD mapper combines branch allocations as:

```text
final_weights_t =
    z1_t * y1_t
    + z2_t * y2_t
    + z3_t * y3_t
    + z4_t * y4_t
```

The existing standalone branch credit treats every branch as a fully invested shadow portfolio. This is only an approximation to its actual contribution:

- a branch with small `z_i` can receive a large standalone return or drawdown signal;
- branch 2 can change the overlap term and therefore alter `z3` and `z4`;
- branch effects include final-portfolio turnover, not only asset return;
- the active RCPO cost belongs to the final combined portfolio.

Counterfactual credit asks what would have happened if one branch had used a neutral action while the other branch decisions were retained or coherently recomputed.

## 2. Configuration And Compatibility

Add explicit branch-credit modes instead of silently changing `standalone`:

```yaml
network:
  branch_credit_mode: standalone_reward_global_cost
```

Phase 2 candidate modes:

```text
counterfactual_open_loop_reward_global_cost
counterfactual_open_loop_reward_cost
counterfactual_prefix_reward_global_cost
counterfactual_prefix_reward_cost
```

Checkpoint metadata must store:

```text
branch_credit_mode
counterfactual_semantics_version
counterfactual_downstream_mode
counterfactual_neutral_action_mode
counterfactual_critic_schema_version
```

Checkpoints from different counterfactual modes are not resumable into one another.

## 3. Neutral Branch Actions

For branch `i`, define a deterministic neutral action:

```text
Gaussian branch logits: all zeros
Dirichlet branch weights: uniform within the branch
inactive branch: no counterfactual is constructed
```

The full CAOSD mapper must be rerun after replacing a branch. Do not directly subtract a padded branch allocation from final weights because changing branch 2 can change downstream CAOSD coefficients.

For actual action:

```text
a_t = (a1_t, a2_t, a3_t, a4_t)
w_actual_t = CAOSD(a_t)
```

For a basic branch-`i` counterfactual:

```text
a_cf_i,t =
    (a1_t, ..., neutral_i, ..., a4_t)

w_cf_i,t = CAOSD(a_cf_i,t)
```

## 4. Counterfactual Reward Implementations

Three practical implementations are possible. They are not theoretically equivalent.

### Option A: Open-Loop Stateful Difference Reward

Replace branch `i` with neutral and keep the other realized branch actions unchanged. Maintain a separate shadow portfolio path for each active branch.

Each shadow path stores:

```text
previous_final_weights_cf_i
portfolio_value_cf_i
running_peak_cf_i
current_drawdown_cf_i
maximum_drawdown_cf_i
previous_turnover_cf_i
```

At each step:

```text
turnover_actual_t =
    L1(w_actual_t - previous_actual_weights)

turnover_cf_i,t =
    L1(w_cf_i,t - previous_final_weights_cf_i)

net_return_actual_t =
    w_actual_risky_t * risky_returns_t
    - transaction_cost_rate * turnover_actual_t

net_return_cf_i,t =
    w_cf_i,risky_t * risky_returns_t
    - transaction_cost_rate * turnover_cf_i,t

reward_actual_t = log(1 + net_return_actual_t)
reward_cf_i,t = log(1 + net_return_cf_i,t)

delta_reward_i,t = reward_actual_t - reward_cf_i,t
```

Advantages:

- easy to implement inside the existing environment;
- uses the complete CAOSD mapping and actual branch mass;
- includes counterfactual transaction costs;
- requires no additional market rollout or RNG stream.

Limitations:

- it is an open-loop alternative path because its future actions are still produced from the actual policy observation;
- for autoregressive policies, holding downstream actions fixed after changing an earlier branch creates an action combination that the policy might not have sampled under the changed prefix;
- it is a heuristic difference reward, not automatically an unbiased control variate.

This is the recommended first Phase 2 reward implementation because it is simple and auditable. It must be labelled `counterfactual_open_loop`, not paper-faithful or unbiased.

### Option B: Prefix-Coherent Counterfactual Reward

For an autoregressive policy, replace branch `i` with neutral and recompute every downstream branch using the modified prefix:

```text
actual:
    a1 -> a2|a1 -> a3|a1,a2 -> a4|a1,a2,a3

counterfactual for branch 2:
    a1 -> neutral_2
       -> a3_cf|a1,neutral_2
       -> a4_cf|a1,neutral_2,a3_cf
```

Use deterministic downstream means for the first implementation. This avoids a second random stream and reduces Monte Carlo variance.

This computation belongs in rollout collection, where the actor model is available. The environment should expose a pure helper that maps a complete branch action into final CAOSD weights and advances a supplied shadow state.

Advantages:

- the counterfactual action is coherent with the autoregressive factorization;
- captures indirect effects of an earlier branch on later branch decisions.

Limitations:

- deterministic downstream actions differ from the stochastic behavior policy;
- actor forward cost increases because later heads are recomputed for each active earlier branch;
- shared network parameters make strict causal interpretation difficult;
- the counterfactual remains a model-dependent heuristic unless the downstream distribution is marginalized.

Use this only after Option A has passed unit and pilot tests.

### Option C: Marginalized COMA-Like Baseline

Approximate the expected outcome over alternative branch actions and downstream responses:

```text
baseline_i(s, prefix) =
    E_{a_i', downstream actions}
    [return | state, prefix before branch i]
```

This can be estimated by several sampled neutral/alternative actions per step.

It is the most principled candidate but is substantially more expensive and complex. It is outside the first V2.6 implementation and should remain a later research extension.

## 5. Counterfactual Cost Implementations

The Phase 1 active cost is benchmark-relative current drawdown:

```text
effective_budget_t =
    max(
        drawdown_budget_floor,
        0.90 * benchmark_current_drawdown_t
    )

cost_actual_t =
    max(0, actual_current_drawdown_t - effective_budget_t)^2
    / drawdown_cost_scale
```

Every counterfactual uses the same online constrained-neutral benchmark and budget. It must not create a separate benchmark.

### Cost Option 1: Global Actual Cost Credit

Keep the Phase 1 approach:

```text
A_cost_i = A_cost_global_actual
```

This is the required control in every Phase 2 reward experiment. It is aligned with global lambda and does not require counterfactual cost critics.

### Cost Option 2: Stateful Counterfactual Cost Difference

For each counterfactual path:

```text
cost_cf_i,t =
    max(0, current_drawdown_cf_i,t - effective_budget_t)^2
    / drawdown_cost_scale

delta_cost_i,t = cost_actual_t - cost_cf_i,t
```

Interpretation:

```text
delta_cost_i > 0:
    the realized branch makes the final portfolio more underwater than neutral

delta_cost_i < 0:
    the realized branch improves relative underwater risk
```

`delta_cost_i` is signed and is only a branch-credit signal. It must never update lambda. Lambda continues to use nonnegative `cost_actual_t` only.

### Cost Option 3: One-Step Counterfactual Cost

Instead of preserving a long counterfactual shadow history, start from the actual pre-step portfolio state and compare only the immediate next state under actual and neutralized branch actions.

```text
actual_next_state = transition(actual_pre_state, w_actual_t)
cf_next_state_i = transition(actual_pre_state, w_cf_i,t)

delta_cost_one_step_i =
    cost(actual_next_state) - cost(cf_next_state_i)
```

Advantages:

- isolates the current action's local effect;
- branch critic needs less hidden counterfactual history;
- easier to hand-check.

Limitations:

- does not represent long-term consequences of repeatedly neutralizing a branch;
- may be sparse when neither next state crosses the budget;
- still needs GAE or a learned critic for delayed effects.

This is a useful ablation and may be more stable than a fully stateful counterfactual cost.

## 6. Autoregressive Bias And Required Labelling

The policy factorization is:

```text
pi(a | s) =
    pi1(a1 | s)
    pi2(a2 | s, a1)
    pi3(a3 | s, a1, a2)
    pi4(a4 | s, a1, a2, a3)
```

For branch 4, neutral replacement has no downstream branch dependency.

For branches 2 and 3, an open-loop counterfactual retains actions whose sampling distribution depended on the replaced action. Therefore open-loop `delta_reward_i` and `delta_cost_i` should be described as heuristic difference credit, not an unbiased policy-gradient baseline.

The current Experiment 2 has inactive branch 1 because `z1 = 0`. If later constraints make branch 1 active, the same downstream-dependency issue applies most strongly to branch 1.

Every report must state which semantics were used:

```text
open-loop fixed downstream actions
prefix-coherent deterministic downstream actions
sampled/marginalized downstream actions
```

## 7. Counterfactual Critic Inputs

Actual observation features alone are insufficient for a stateful counterfactual critic. The target depends on the counterfactual path's history.

Keep the actor observation from Phase 1. Construct critic-only branch context for each active branch:

```text
counterfactual_current_weights_i
counterfactual_previous_turnover_i
counterfactual_portfolio_value_relative_to_actual_i
counterfactual_current_drawdown_i
counterfactual_max_drawdown_i
counterfactual_drawdown_gap_i
episode_progress
```

Recommended architecture:

```text
shared_market_features = actor_backbone(actual_observation)

branch_critic_input_i = concat(
    shared_market_features,
    counterfactual_context_i
)

V_delta_reward_i = branch_reward_critic_i(branch_critic_input_i)
V_delta_cost_i = branch_cost_critic_i(branch_critic_input_i)
```

The actor must not receive counterfactual future information. All counterfactual context is generated online from current and past realized returns only.

For one-step counterfactual cost, the critic can use actual pre-step state plus branch identity and does not require persistent counterfactual portfolio history.

## 8. Advantages And Actor Objectives

### Counterfactual Reward With Global Cost

Recommended first Phase 2 objective:

```text
A_delta_reward_i =
    GAE(delta_reward_i, V_delta_reward_i)

A_i_PPO = A_delta_reward_i

A_i_RCPO =
    A_delta_reward_i
    - lambda * A_cost_global_actual
```

### Fully Counterfactual Reward And Cost

Later candidate:

```text
A_delta_cost_i =
    GAE(delta_cost_i, V_delta_cost_i)

A_i_RCPO =
    A_delta_reward_i
    - lambda * A_delta_cost_i
```

Normalize the selected advantage separately per active branch, matching the current standalone branch pipeline.

The policy loss is:

```text
L_policy = -mean_t sum_active_i min(
    ratio_i,t * normalized_A_i,t,
    clipped_ratio_i,t * normalized_A_i,t
)
```

Do not multiply the counterfactual objective by `z_i`. The full CAOSD counterfactual already includes actual branch mass, dynamic coefficients, and indirect effects. A branch whose replacement does not affect final weights should naturally have near-zero difference targets.

For diagnostics, continue logging actual `z_i` even though it is not a loss multiplier in counterfactual modes.

## 9. Lambda And Feasibility Remain Global

Counterfactual differences never determine feasibility.

Lambda update:

```text
lambda_gap =
    mean(actual_final_portfolio_cost)
    - mean(dynamic_alpha)

lambda_next =
    max(0, lambda_before + selected_lr * lambda_gap)
```

This occurs once per rollout, as defined in Phase 1.

Validation feasibility, robust 80% fallback selection, and all reported constraint metrics use only actual final portfolio costs. Counterfactual cost can be negative and must not appear in the active constraint-cost field.

## 10. Logging

Extend the Phase 1 compact logs with active-branch aggregates:

```text
branch_actual_reward_mean_i
branch_counterfactual_reward_mean_i
branch_delta_reward_mean_i
branch_delta_reward_std_i

branch_actual_cost_mean_i
branch_counterfactual_cost_mean_i
branch_delta_cost_mean_i
branch_delta_cost_std_i

branch_delta_reward_critic_ev_i
branch_delta_cost_critic_ev_i
branch_lambda_delta_cost_adv_ratio_i

counterfactual_weight_l1_distance_mean_i
counterfactual_turnover_difference_mean_i
counterfactual_drawdown_difference_mean_i
```

Add consistency diagnostics:

```text
counterfactual_zero_effect_rate_i
counterfactual_nonfinite_count
counterfactual_mapping_failure_count
```

Inactive branches use `null`. Do not store step-level counterfactual arrays in `metrics.jsonl`; aggregate them during rollout collection.

## 11. Implementation Locations

### `simplex.py`

- expose branch splitting and neutral branch replacement helpers;
- provide a pure full-action-to-final-weights mapping for actual and counterfactual actions;
- preserve numerical feasibility checks.

### `models.py`

- preserve the Phase 1 actor interfaces;
- add optional counterfactual-context branch critics;
- expose deterministic downstream branch reconstruction for prefix-coherent mode;
- save counterfactual architecture metadata.

### `env.py`

- maintain open-loop counterfactual shadow states;
- apply actual and counterfactual transitions to the same realized market return;
- include each path's own previous weights and transaction costs;
- return aggregated branch delta reward/cost inputs in `info`;
- never use future returns.

### `rollouts.py`

- collect branch actual/counterfactual/delta streams;
- construct critic-only counterfactual context;
- compute delta-reward and optional delta-cost GAE;
- aggregate diagnostics without storing unnecessary full histories in metrics.

### `algorithms/ppo.py` And `algorithms/rcpo.py`

- select the configured branch advantage stream;
- omit `z_i` multiplication only in counterfactual modes;
- keep Phase 1 non-negative KL and pre-update rejection;
- keep one global lambda based on actual portfolio cost.

### `trainer.py` And `evaluation.py`

- persist and validate counterfactual mode metadata;
- keep validation/checkpoint feasibility based on actual portfolio only;
- add counterfactual diagnostics to training logs, not ordinary final-return plots.

## 12. Tests

### Mapping And State

- Neutral replacement reruns the complete CAOSD mapping.
- Actual and counterfactual weights are nonnegative, sum to one, and satisfy hard constraints.
- Branch 2 replacement correctly changes downstream `z3/z4` when overlap allocation changes.
- Every shadow path resets from the constrained-neutral initial portfolio.
- Counterfactual paths use the same current market return and no future information.
- Each shadow path uses its own previous weights for turnover.

### Reward

- If actual branch action equals neutral, `delta_reward_i` is zero within tolerance.
- Hand-calculated returns and transaction costs match the open-loop formula.
- A beneficial branch produces positive difference reward in a deterministic example.
- An inactive or no-effect branch produces no meaningful actor gradient.

### Cost

- Actual and counterfactual current drawdowns are hand checked.
- `delta_cost_i` can be positive, zero, or negative.
- Lambda uses actual nonnegative cost only.
- Validation feasibility ignores counterfactual difference cost.

### Autoregressive Behavior

- Open-loop mode preserves realized downstream actions and is labelled heuristic.
- Prefix mode recomputes all downstream branches from the changed prefix.
- Branch 4 open-loop and prefix modes agree when no other implementation detail differs.
- Deterministic prefix recomputation uses no additional RNG stream.

### Critics And Optimization

- Stateful counterfactual critics receive their own online context.
- Counterfactual reward mode does not multiply policy loss by `z_i`.
- Global-cost and counterfactual-cost modes select the intended cost advantage.
- Joint/branch KL safety remains unchanged from Phase 1.

### Compatibility

- Counterfactual modes reject incompatible Phase 1/V2.5 checkpoints.
- Historical evaluation remains available for old checkpoints.
- Logging schema distinguishes open-loop, prefix, and global-cost variants.

## 13. Experimental Sequence

Use the same seed set, market configuration, validation branches, and evaluation branches for every comparison.

Recommended sequence:

1. Phase 1 control: standalone reward + global actual cost.
2. Open-loop counterfactual reward + global actual cost.
3. Prefix-coherent counterfactual reward + global actual cost.
4. Open-loop counterfactual reward + one-step counterfactual cost.
5. Open-loop counterfactual reward + stateful counterfactual cost.
6. Prefix-coherent reward/cost only if earlier variants provide clear benefit.

Do not begin with fully counterfactual reward and cost. Reward-credit benefit must be established before adding counterfactual risk attribution.

Use at least three seeds for the pilot and five seeds for the final comparison where practical.

## 14. Acceptance Criteria

A counterfactual variant is worth retaining only if it improves over the Phase 1 control on more than a single best checkpoint:

- hard allocation feasibility remains exact;
- mean future relative wealth improves or remains statistically comparable;
- validation-to-test degradation is reduced;
- RCPO robust-feasible checkpoint frequency does not decline;
- branch critic explained variance improves over standalone branch critics;
- counterfactual reward/cost scales remain finite and auditable;
- policy turnover does not increase materially without compensating return;
- benefit appears across multiple seeds and future branches;
- open-loop and prefix results are reported under their correct theoretical interpretation.

If counterfactual credit does not outperform Phase 1, retain the simpler standalone-reward/global-cost design and present counterfactual attribution as a negative or exploratory result.
