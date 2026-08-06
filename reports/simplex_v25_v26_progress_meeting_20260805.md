# Constraint-Aware Portfolio RL: V2.5 to V2.6 Progress Update

**Meeting date:** 5 August 2026  
**Project:** Constraint-Aware Simplex Decomposition for Robust Portfolio Reinforcement Learning  
**Status:** V2.6 Phase 1 completed and evaluated; Phase 2 counterfactual credit is the next planned experiment.

---

## 1. Meeting Goal

### Key Words

hard allocation feasibility | relative drawdown control | 8-asset environment | V2.5 to V2.6 | generalization

### Explanation

This meeting reviews the transition from V2.5 to V2.6. The objective was not only to increase validation return. It was to make the constrained portfolio-learning problem harder, make the risk-control signal observable to the policy, and make PPO/RCPO training behavior auditable.

The questions are:

1. Do the V2.6 corrections improve performance on unseen continuation markets relative to the constrained-neutral baseline?
2. Do hard Simplex Decomposition constraints and soft RCPO constraints remain distinguishable in the more difficult environment?
3. What remains unresolved before introducing a more ambitious branch-credit method?

---

## 2. Project Context

### Key Words

CAOSD | PPO | RCPO | hard allocation constraints | soft risk constraint

### Explanation

The project combines two complementary ideas:

- **Simplex Decomposition (CAOSD):** maps branch actions to a final portfolio that satisfies two allocation constraints by construction.
- **RCPO:** optimizes return subject to a learned cost through one Lagrange multiplier, lambda.

For simplex policies, allocation feasibility is hard: the final portfolio satisfies the allocation rules up to numerical tolerance. RCPO is used separately to control portfolio risk. For non-simplex baselines, RCPO learns allocation violations, or a joint allocation-plus-drawdown cost, as soft penalties.

The common deterministic reference is the **constrained-neutral CAOSD baseline**, not arithmetic 1/N equal weight. It is the neutral feasible allocation under the same constraints.

---

## 3. V2.5: A More Difficult Portfolio Environment

### Key Words

5 to 8 risky assets | cash plus 8 assets | regimes | momentum | overlapping allocation groups

### Explanation

Earlier experiments used cash plus five risky assets. V2.5 moves to **cash plus eight risky assets**, while retaining regime-dependent returns, momentum, correlations, transaction costs, eight independent training markets, and multi-branch validation/test continuations.

The larger universe removes the earlier shortcut in which one asset could conveniently satisfy both allocation groups while also having strong expected return. The active allocation requirements are:

    V1 = {Asset 1, Asset 4, Asset 6}, minimum total weight = 0.50
    V2 = {Asset 2, Asset 4, Asset 7}, minimum total weight = 0.40

Asset 4 is the only overlap. Because 0.50 + 0.40 <= 1, mandatory overlap mass is zero in this configuration (z1 = 0), so the inactive singleton intersection branch is excluded from actor training. The other active branches must still allocate across different subsets while satisfying both requirements.

The market signal is also less one-dimensional: Assets 1-2 are stronger in low-volatility conditions, Assets 6-8 are more defensive in high-volatility conditions, and the policy must infer this from trailing information rather than observe a regime label.

### Discussion Point

The 8-asset setting is a meaningful increase in difficulty, but it remains a controlled synthetic environment. Its role is to identify algorithmic failure modes before moving to historical-market experiments.

[Figure Placeholder: V2.5 environment, asset groups, and CAOSD constraint sets]

---

## 4. What Was Unconvincing in V2.5

### Key Words

path-dependent cost | partial observability | lambda cadence | KL safety | branch credit

### Explanation

V2.5 produced useful policies, but its training core still had several weaknesses:

- The active maximum-drawdown cost remained active even after recovery, making it difficult to attribute the cost to current actions.
- The observation did not fully expose the state that determined the drawdown cost.
- Lambda and KL diagnostics were not yet sufficiently auditable.
- Standalone branch return credit was only an approximation to a branch's effect on the final CAOSD portfolio.
- A return-best checkpoint could look strong on fixed validation branches while carrying a substantial risk-cost violation on new branches.

V2.6 therefore changes training correctness, state observability, and diagnostics before changing the CAOSD feasibility mapping again.

---

## 5. V2.6 Phase 1: Relative Current Drawdown and Training-Core Corrections

### Key Words

relative current drawdown | online benchmark | dynamic alpha | one lambda update per rollout | KL rejection

### New Risk Cost

V2.6 replaces the active maximum-drawdown occupancy cost with a **relative current-drawdown** cost. Maximum drawdown remains an evaluation metric, but the active training cost can return to zero after recovery.

    agent_current_drawdown_t =
        (agent_running_peak_t - agent_portfolio_value_t)
        / agent_running_peak_t

    benchmark_current_drawdown_t =
        (benchmark_running_peak_t - benchmark_portfolio_value_t)
        / benchmark_running_peak_t

    budget_t = max(0.05, 0.90 * benchmark_current_drawdown_t)

    violation_t = max(0, agent_current_drawdown_t - budget_t)
    constraint_cost_t = violation_t^2 / 0.10

The benchmark is the same online constrained-neutral CAOSD portfolio on the same realized market return. There is no look-ahead and no second environment rollout.

### Dynamic Alpha and Lambda

The allowable average cost is expressed in the same units as the cost:

    alpha_t = ((0.05 * budget_t)^2) / 0.10
    rollout_alpha = mean_t(alpha_t)

    lambda_gap = mean_t(constraint_cost_t) - rollout_alpha
    lambda_next = max(0, lambda_before + selected_learning_rate * lambda_gap)

Lambda is updated **once per rollout**, after PPO/RCPO optimization. The current rollout uses lambda_before; lambda_next applies to the next rollout. The number of PPO epochs or minibatches therefore cannot accidentally change the number of multiplier updates.

### Other Phase 1 Corrections

- Added agent/benchmark current and maximum drawdown, budget, drawdown gap, and episode progress to the observation.
- Corrected the simplex actor-gradient path and restarted V2.6 experiments after the fix.
- Used standalone branch return advantages with the **global final-portfolio cost advantage** for RCPO. One global lambda remains responsible for real final-portfolio risk.
- Replaced the signed sampled KL diagnostic with a non-negative estimator and reject a minibatch before its optimizer step if it exceeds target KL.
- Split compact training metrics from complete validation metrics, strengthening resume and plotting behavior.
- Clarified checkpoints: checkpoint_best_return.pt is selected by validation return; checkpoint_best_feasible.pt ranks first by feasible-branch rate and then by validation return; checkpoint_last.pt is recovery only.

[Figure Placeholder: Online relative current-drawdown calculation]

[Figure Placeholder: V2.6 training-core changes and diagnostics]

---

## 6. Evaluation Protocol

### Key Words

shared future markets | 20 branches | clean evaluation | constrained-neutral baseline | relative wealth

### Explanation

For each version, policies are evaluated deterministically on the **same 20 future continuation markets**. Evaluation uses clean realised returns. The V2.6 comparison uses checkpoint_best_return.pt; test branches never influence checkpoint selection.

Return presentation is relative wealth, not a simple subtraction of returns:

    relative_wealth_t =
        portfolio_wealth_t / constrained_neutral_wealth_t - 1

For example, +8.06% means that, averaged across the 20 one-year future markets, the policy ended with 8.06% more wealth than the constrained-neutral baseline.

---

## 7. Shared 20-Market Results: V2.5 Versus V2.6

### Key Words

unseen future markets | five of six improved | RCPO Dirichlet strongest | risk feasibility remains open

| Policy | V2.5 relative wealth | V2.6 relative wealth | Change | V2.5 win rate | V2.6 win rate | V2.5 mean max DD | V2.6 mean max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO Gaussian | +5.84% | +4.83% | -1.01 pp | 85% | 75% | 17.07% | 17.99% |
| PPO Dirichlet | +4.06% | +5.58% | +1.52 pp | 80% | 70% | 16.93% | 17.41% |
| RCPO Gaussian | +3.88% | +4.61% | +0.73 pp | 60% | 70% | 18.72% | 19.14% |
| RCPO Dirichlet | +4.54% | **+8.06%** | **+3.52 pp** | 80% | **90%** | 16.76% | **16.73%** |
| RCPO allocation penalty | +5.79% | +5.86% | +0.07 pp | 75% | 80% | 17.53% | 16.99% |
| RCPO allocation + drawdown | +4.78% | +5.46% | +0.68 pp | 60% | 75% | 17.41% | 17.43% |
| Constrained-neutral baseline | 0.00% | 0.00% | - | - | - | 16.70% | 16.70% |

### Interpretation

- **Five of six methods improved** on the shared unseen markets after the V2.6 changes. PPO Gaussian is the exception: both relative wealth and drawdown worsened slightly.
- **RCPO Dirichlet is the strongest V2.6 return-best model:** +8.06% relative wealth, 90% branch win rate, and mean maximum drawdown almost equal to the constrained-neutral baseline.
- **RCPO Gaussian remains economically questionable** despite low turnover (1.91%): it has the highest V2.6 mean drawdown (19.14%) among learned policies and only 15% active-risk feasibility across branches.
- The soft **allocation-only RCPO** remains competitive in return (+5.86%) and improves drawdown, but its active allocation-cost feasibility is still 0%. It reduces violations without enforcing them.
- All simplex policies retain **hard allocation feasibility by construction**. The feasibility rates in this report refer instead to their active RCPO risk-cost condition.

[Figure Placeholder: V2.5 cumulative return comparison]

Suggested source: evaluation/section9_simplex_v2.5_policy_comparison/section9_cumulative_return_comparison.png

[Figure Placeholder: V2.6 cumulative return comparison]

Suggested source: evaluation/section9_simplex_v2.6_policy_comparison/section9_cumulative_return_comparison.png

[Figure Placeholder: V2.5 versus V2.6 maximum drawdown and turnover]

Suggested sources:

- evaluation/section9_simplex_v2.5_policy_comparison/section9_max_drawdown_comparison.png
- evaluation/section9_simplex_v2.6_policy_comparison/section9_max_drawdown_comparison.png
- evaluation/section9_simplex_v2.6_policy_comparison/section9_turnover_comparison.png

---

## 8. Validation-to-Test Gap: Important Limitation

### Key Words

selection bias | different future branches | finite-sample uncertainty | not a final claim

Every return-best checkpoint was selected repeatedly on the same fixed 10 validation continuation markets. The shared evaluation uses a different set of 20 test continuation markets. The selected validation result is therefore expected to be optimistic relative to the test result. The V2.5 results use the historical `checkpoint_best.pt`; the V2.6 results use the explicit `checkpoint_best_return.pt`.

### V2.5 Selected-Checkpoint Gap

| Policy | Validation relative wealth | Shared-test relative wealth | Test minus validation |
|---|---:|---:|---:|
| PPO Gaussian | +8.83% | +5.84% | -2.99 pp |
| PPO Dirichlet | +6.84% | +4.06% | -2.78 pp |
| RCPO Gaussian | +10.17% | +3.88% | -6.29 pp |
| RCPO Dirichlet | +6.92% | +4.54% | -2.38 pp |
| RCPO allocation penalty | +10.23% | +5.79% | -4.44 pp |
| RCPO allocation + drawdown | +9.46% | +4.78% | -4.68 pp |

### V2.6 Selected-Checkpoint Gap

| Policy | Validation relative wealth | Shared-test relative wealth | Test minus validation |
|---|---:|---:|---:|
| PPO Gaussian | +9.88% | +4.83% | -5.05 pp |
| PPO Dirichlet | +9.37% | +5.58% | -3.79 pp |
| RCPO Gaussian | +11.20% | +4.61% | -6.59 pp |
| RCPO Dirichlet | +8.90% | +8.06% | **-0.84 pp** |
| RCPO allocation penalty | +9.28% | +5.86% | -3.42 pp |
| RCPO allocation + drawdown | +9.68% | +5.46% | -4.22 pp |

### Change in Validation-to-Test Gap

| Policy | V2.5 gap | V2.6 gap | Interpretation |
|---|---:|---:|---|
| PPO Gaussian | -2.99 pp | -5.05 pp | Future performance and gap both worsened. |
| PPO Dirichlet | -2.78 pp | -3.79 pp | Future return improved, but the selection gap widened. |
| RCPO Gaussian | -6.29 pp | -6.59 pp | Large gap remains; this is the least convincing RCPO result. |
| RCPO Dirichlet | -2.38 pp | **-0.84 pp** | Both future return and transfer stability improved strongly. |
| RCPO allocation penalty | -4.44 pp | -3.42 pp | Small return improvement and better transfer stability. |
| RCPO allocation + drawdown | -4.68 pp | -4.22 pp | Small return improvement and slightly better transfer stability. |

V2.6 is better than V2.5 in shared future performance overall, but its validation-to-test transfer is mixed rather than uniformly improved. The clearest positive result is RCPO Dirichlet: it has the highest shared-test relative wealth and the smallest gap. The remaining gaps show that repeated checkpoint selection on a small fixed validation set remains a material source of optimism. Evidence is still limited to one training seed and a finite synthetic market set.

---

## 9. Current Conclusions

### Key Words

better training core | Dirichlet RCPO promising | hard constraints work | soft constraints incomplete | generalization remains open

1. The V2.6 training-core revision was worthwhile. It improved five of the six shared-test comparisons, strengthened diagnostics, and made the path-dependent risk objective visible to the policy.
2. Hard CAOSD constraints work as intended: simplex portfolios meet allocation requirements without relying on a penalty multiplier.
3. The V2.6 RCPO Dirichlet policy is the most promising current result, with the best relative wealth, near-baseline drawdown, and a high future-branch win rate.
4. A high-return checkpoint_best_return.pt is not automatically a risk-feasible policy. The separate checkpoint_best_feasible.pt is necessary for honest RCPO reporting.
5. Soft RCPO allocation constraints are useful comparison baselines, but their zero active-feasibility rates show why hard Simplex Decomposition is valuable when feasibility must be guaranteed.

---

## 10. Next Step: V2.6 Phase 2 Counterfactual Branch Credit

### Key Words

counterfactual reward | branch responsibility | CAOSD remapping | global lambda | careful pilot

### Motivation

V2.6 Phase 1 keeps a deliberate approximation: each active branch learns from a standalone return path, although its true effect depends on its CAOSD mass, the other branch actions, and final-portfolio transaction costs. This can slow or misdirect credit assignment.

Phase 2 will test a counterfactual difference signal. For branch i, compare the realised complete action with a complete action where only that branch is replaced by its neutral action:

    w_actual = CAOSD(a_1, ..., a_i, ..., a_4)
    w_cf_i   = CAOSD(a_1, ..., neutral_i, ..., a_4)

    delta_reward_i = reward_actual - reward_cf_i

CAOSD must be rerun for every counterfactual action. This preserves the actual branch mass, allocation constraints, and transaction-cost effect; directly subtracting a branch allocation would be incorrect.

### Planned Order

1. Keep V2.6 Phase 1 as the control: relative current drawdown, dynamic alpha, global final-portfolio cost, and the current evaluation protocol.
2. Start with **open-loop stateful counterfactual reward plus global actual cost**. This is the simplest auditable version and does not change the one-global-lambda rule.
3. Compare it with the Phase 1 standalone-reward control using the same seeds, markets, architecture, and checkpoint criteria.
4. Only if the reward experiment is stable, test counterfactual cost credit as a separate ablation. Lambda will still use the non-negative actual final-portfolio cost, never branch counterfactual cost.
5. Label open-loop autoregressive results honestly: downstream branch actions were sampled under the actual prefix, so this is a heuristic difference credit rather than an unbiased causal estimator.

### Success Criteria

- Preserve hard allocation feasibility and stable KL behavior.
- Improve shared-future performance or return retention without excessive concentration or turnover.
- Increase the rate of branches satisfying the active risk-cost target for RCPO models.
- Report both return-best and feasible-best checkpoints; do not select a method from validation return alone.

[Figure Placeholder: Phase 2 counterfactual branch-credit pipeline]

---

## 11. Questions for Discussion

1. Is relative current drawdown the most appropriate intermediate risk signal before using a more realistic drawdown or tail-risk constraint?
2. Should the next comparison prioritize RCPO Dirichlet, or retain all six methods despite higher compute cost?
3. Is open-loop counterfactual reward a sufficient first research contribution if its limitations are stated clearly?
4. What level of multi-seed and multi-branch evidence is needed before moving from synthetic to historical market data?

---

## One-Minute Summary

In this stage, I increased the environment from five to eight risky assets and redesigned the allocation constraints to make simple concentration shortcuts harder. V2.5 showed that the methods could learn, but the risk cost was history dependent and not fully represented in the observation, while lambda and KL behavior needed stronger diagnostics. In V2.6, I changed the active RCPO cost to benchmark-relative current drawdown, added the relevant state to the observation, made alpha and lambda consistent with that cost, corrected the simplex actor-gradient path, and improved checkpoint and optimizer auditing. On 20 shared unseen future markets, five of six methods improved; RCPO Dirichlet is currently strongest at +8.06% relative wealth versus the constrained-neutral baseline with a 90% win rate. However, validation-to-test gaps and imperfect RCPO feasibility show that the problem is not solved. The next step is counterfactual branch credit, so each CAOSD branch can learn from a more faithful estimate of its contribution to return and risk.

---

## Data and Figure Sources

- V2.5 summary: evaluation/section9_simplex_v2.5_policy_comparison/section9_comparison_summary.json
- V2.6 summary: evaluation/section9_simplex_v2.6_policy_comparison/section9_comparison_summary.json
- V2.5 figures: evaluation/section9_simplex_v2.5_policy_comparison/
- V2.6 figures: evaluation/section9_simplex_v2.6_policy_comparison/
- Phase 1 design: reports/v2.6/v2_6_phase1_relative_current_drawdown_training_core.md
- Phase 2 design: reports/v2.6/v2_6_phase2_counterfactual_branch_credit.md


