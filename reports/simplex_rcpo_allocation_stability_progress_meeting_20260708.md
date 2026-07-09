# Progress Meeting: RCPO Allocation Penalties And Simplex Policy Stability

**Meeting date:** 8 July 2026  
**Project:** Constraint-Aware Simplex Decomposition for Robust Portfolio Reinforcement Learning

---

## 1. Meeting Goal

**Key Words**

- Three approaches to allocation constraints
- Pure RCPO soft penalty
- Simplex hard feasibility
- Gaussian KL early stopping
- Dirichlet training stability


**Explanation**

Since the previous meeting, the project has progressed in two directions.

First, I implemented a pure RCPO solution for the same allocation constraints used by Simplex Decomposition. This creates a direct comparison between learning constraints through a soft penalty and satisfying them by construction.

Second, I continued the simplex policies for much longer training. This exposed two architecture-specific problems:

- The Gaussian KL problem is much smaller than in v2.2, but it still rises again late in long training.
- Dirichlet policies remain numerically stable, but validation performance can still drift away from a previously learned policy.


---

## 2. Starting Point From The Previous Meeting

**Key Words**

- CAOSD hard allocation constraints
- Four autoregressive branches
- Standalone branch credit
- Gaussian logits versus Dirichlet weights
- One global RCPO multiplier

**Explanation**

The previous implementation combined Simplex Decomposition with standalone branch credit assignment.

For two allocation constraints, the final portfolio is composed from four branch portfolios:

```text
w = z1 y1 + z2 y2 + z3 y3 + z4 y4
```

Each branch receives its own return and drawdown-cost advantage, weighted by its actual CAOSD contribution `z_i`. RCPO still uses one global Lagrange multiplier, updated from the final combined portfolio constraint cost.

Two policy distributions were being compared:

```text
Gaussian:  branch logits -> softmax -> branch allocation
Dirichlet: concentration vector -> direct branch allocation
```

---

## 3. Three Solutions To The Constraint Problem

**Key Words**

- Soft allocation penalty
- Hard simplex feasibility
- Hard allocation plus dynamic risk control

**Explanation**

The project now contains three conceptually distinct solutions:

1. **Pure RCPO allocation penalty**
   - Standard softmax action.
   - Allocation feasibility is learned through a Lagrange penalty.
   - Violations may remain.

2. **Pure Simplex Decomposition**
   - Allocation constraints are guaranteed by the action mapping.
   - PPO optimizes return without an allocation penalty.

3. **Simplex Decomposition plus RCPO**
   - Allocation constraints remain hard.
   - RCPO controls a separate path-dependent risk objective: benchmark-relative maximum drawdown.

This separation makes the comparison precise. The first two methods solve the same allocation-constraint problem in different ways. The third method asks whether RCPO can add useful dynamic risk control after allocation feasibility is already guaranteed.

**Figure Placeholder: Three-method comparison**

Suggested content:

```text
Softmax + allocation RCPO
Simplex + PPO
Simplex + drawdown RCPO
```

---

## 4. New Baseline: Pure RCPO For Allocation Constraints

**Key Words**

- No Simplex Decomposition
- No Dirichlet
- Flat Gaussian softmax policy
- Allocation violations as RCPO cost
- Soft feasibility, not guaranteed feasibility

**Explanation**

I added a pure RCPO baseline that uses a standard softmax portfolio policy. It does not use CAOSD or any simplex branch architecture.

The baseline is trained against the same two allocation constraints used by the simplex experiments:

```text
Constraint 1: assets [1, 3, 5], minimum combined weight = 0.55
Constraint 2: assets [4, 5],    minimum combined weight = 0.55
```

The overlap through asset 5 makes this a useful test of whether a learned penalty can coordinate both requirements without a hard feasibility mapping.

---

## 5. How The Pure RCPO Allocation Cost Works

**Key Words**

- Normalized violation
- Squared cost
- Cost scaling
- Fixed alpha tolerance
- Adaptive lambda

**Explanation**

For each allocation constraint:

```text
violation_i =
    max(minimum_weight_i - actual_group_weight_i, 0)
    / minimum_weight_i

violation_cost_i = violation_i^2
```

The final RCPO allocation cost is:

```text
raw_allocation_cost =
    violation_cost_1 + violation_cost_2

constraint_cost =
    raw_allocation_cost / allocation_constraint_cost_scale
```

Current settings:

```yaml
allocation_constraint_cost_scale: 20.0

rcpo:
  alpha: 0.0005
  lambda_lr_up: 0.0015
  lambda_lr_down: 0.03
```

The actor uses:

```text
A_combined = A_return - lambda * A_allocation_cost
```

The multiplier increases when average cost exceeds `alpha` and relaxes more quickly when cost is below `alpha`.

**Important Interpretation**

RCPO can reduce constraint violations, but it does not guarantee zero violation. Simplex Decomposition provides hard feasibility: every mapped action satisfies the allocation constraints, even before training.


---

## 6. What Longer v2.2 Training Revealed

**Key Words**

- Short runs hid optimization problems
- Gaussian updates were mostly truncated
- Dirichlet validation could drift
- Best checkpoint differs from latest checkpoint

**Evidence**

The v2.2 runs were continued to approximately `84,000` updates. The long horizon revealed behavior that was not obvious in the early experiments.

| Run | KL Early-Stop Rate | Mean Optimizer Steps | Best Validation Excess | Latest Validation Excess |
|---|---:|---:|---:|---:|
| PPO Gaussian v2.2 | `93.7%` | `6.64 / 16` | `+10.61%` | `+2.74%` |
| RCPO Gaussian v2.2 | `95.8%` | `6.22 / 16` | `+6.63%` | `+6.16%` |
| PPO Dirichlet v2.2 | `3.1%` | `15.85 / 16` | `+7.59%` | `+7.59%` |
| RCPO Dirichlet v2.2 | `9.1%` | `15.53 / 16` | `+6.06%` | `+3.70%` |

**Interpretation**

The Gaussian runs were nominally configured for:

```text
4 epochs x 4 minibatches = 16 optimizer steps per update
```

The Gaussian runs completed only about six to seven optimizer steps on average. The Dirichlet runs usually completed nearly all 16 steps, but validation performance could still drift away from the best checkpoint.

This separates two problems:

```text
Gaussian:  optimization is repeatedly truncated by KL protection.
Dirichlet: optimization completes, but the learned distribution can still drift.
```

**Figure Placeholder: v2.2 validation-score history**

Source:

- `evaluation/section9_simplex_v2.2_policy_comparison/section9_validation_score_history.png`

---

## 7. Problem 1: Gaussian KL Early Stopping

**Key Words**

- Joint autoregressive policy
- Excessive update size
- Incomplete PPO epochs
- Effective training budget

**Explanation**

PPO stops the remaining minibatch updates when the approximate KL divergence exceeds `target_kl`. This protects the policy from a destructive update, but an early-stop rate above `93%` means that the planned optimization is rarely completed.

For the autoregressive Gaussian policy, several effects accumulate:

- Later branches depend on earlier branch actions.
- The joint policy change contains contributions from multiple branches.
- A change in an early branch also changes the input to later branches.
- The Gaussian log-probability can change substantially even when the final allocation changes less dramatically.

The issue is therefore not that KL early stopping is harmful. The issue is that the policy attempts an excessive change almost every update, so much of the configured optimization budget is unused.

**Figure Placeholder: Gaussian optimizer steps and KL early-stop rate**

Suggested content:

- v2.2 versus v2.3 early-stop percentage.
- Mean completed optimizer steps out of 16.
- KL by branch from `metrics.jsonl`.

---

## 8. Problem 2: Dirichlet Training Instability

**Key Words**

- Direct simplex distribution
- Concentration controls exploration
- Sharp policies
- Validation peaks can be temporary
- Low KL does not guarantee stability

**Explanation**

A Dirichlet branch samples portfolio weights directly:

```text
y_i ~ Dirichlet(alpha_i)
```

The concentration vector controls both the mean allocation and the dispersion.

- Small total concentration produces broad, variable samples.
- Large concentration produces a nearly deterministic policy.
- Strong imbalance between components creates highly concentrated asset preferences.

Long training showed that low KL early-stop rates do not guarantee stable validation performance. A Dirichlet policy can complete nearly every optimizer step while its concentration geometry gradually changes and its out-of-sample policy drifts.

**Late v2.2 Recent Mean Entropy By Branch**

| Branch | Branch Structure | PPO Dirichlet v2.2 | RCPO Dirichlet v2.2 |
|---|---|---:|---:|
| Branch 1 | Singleton intersection | `0.00` | `0.00` |
| Branch 2 | Three-asset simplex | `-3.53` | `-2.67` |
| Branch 3 | Two-asset simplex | `-1.14` | `-1.05` |
| Branch 4 | Six-asset full universe | `-9.50` | `-9.37` |

Branch 1 has entropy zero because it contains only one asset. Its action is fixed at `[1]`, so there is no distribution to explore.

For continuous distributions, negative differential entropy is valid. More negative entropy generally indicates a sharper distribution, but raw values should be compared within the same branch because branch dimensions differ. The clearest evidence is branch 4: both v2.2 policies became extremely sharp over the full asset universe.

This supports the interpretation that the Dirichlet policy can become too deterministic or over-specialized, especially in its highest-dimensional branch.

**Figure Placeholder: v2.2 Dirichlet entropy by branch**

Suggested content:

- Four entropy trajectories for PPO Dirichlet.
- Four entropy trajectories for RCPO Dirichlet.
- Branch 1 shown as a fixed zero reference line.

---

## 9. v2.3 Intervention

**Key Words**

- Lower learning rate
- Tighter PPO clipping
- Same target KL
- Lower Dirichlet concentration ceiling
- More controlled comparison

Compared with v2.2:

```text
learning rate:                 0.0002 -> 0.0001
PPO clip range:                0.15   -> 0.10
target_kl:                     0.02   -> 0.02 (unchanged)
Dirichlet initial alpha:       2.0    -> 1.5
Dirichlet minimum alpha:       0.3    -> 0.5
Dirichlet maximum alpha:       12.0   -> 8.0
```

**Design Rationale**

- Lower learning rate reduces the size of each policy update.
- Tighter clipping prevents a minibatch from exploiting a large probability-ratio change.
- Raising the minimum concentration avoids extremely sparse, numerically fragile components.
- Lowering the maximum concentration limits premature determinism.
- Keeping `target_kl = 0.02` allows a direct test of whether update size, rather than the threshold itself, was the main problem.

---

## 10. Did v2.3 Fix Gaussian KL Early Stopping?

**Key Words**

- Still far better than v2.2
- PPO Gaussian: `14.9%` overall
- RCPO Gaussian: `10.9%` overall
- Most optimizer steps still complete
- Recent rate is climbing again

**Evidence At Approximately 33,000-38,000 Updates**

| Run | KL Early-Stop Rate | Mean Optimizer Steps | Recent 1,000-Update Early-Stop Rate |
|---|---:|---:|---:|
| PPO Gaussian v2.3 | `14.9%` | `15.12 / 16` | `22.3%` |
| RCPO Gaussian v2.3 | `10.9%` | `15.33 / 16` | `18.4%` |

For comparison:

```text
PPO Gaussian v2.2:  93.7% early stopped
RCPO Gaussian v2.2: 95.8% early stopped
```

The Gaussian intervention still addresses the main v2.2 failure mode: almost all planned optimizer steps now complete. However, the long run shows that KL pressure is not fully gone. The recent 1,000-update rate has risen again, so the Gaussian policy is healthier than before, but not completely settled.

**Figure Placeholder: Gaussian KL early-stop comparison**

Suggested content:

- Grouped bars for v2.2 and v2.3 early-stop rates.
- Completed optimizer steps for PPO Gaussian and RCPO Gaussian.

---

## 11. Did v2.3 Stabilize Dirichlet?

**Key Words**

- KL remains fully controlled
- PPO Dirichlet keeps more of its early best than before
- RCPO Dirichlet drifts later in training
- Entropy still differs strongly by branch
- Stability problem remains unresolved

**Evidence At Approximately 33,000-38,000 Updates**

| Run | KL Early-Stop Rate | Best Validation Excess | Best Update | Latest Validation Excess | Drop From Best |
|---|---:|---:|---:|---:|---:|
| PPO Dirichlet v2.3 | approximately `0.0%` | `+4.83%` | `2,129` | `+3.99%` | `0.84%` |
| RCPO Dirichlet v2.3 | `0.4%` | `+6.44%` | `25,799` | `+3.86%` | `2.58%` |

**Recent Mean Entropy By Branch**

| Branch | PPO Dirichlet v2.2 | PPO Dirichlet v2.3 | RCPO Dirichlet v2.2 | RCPO Dirichlet v2.3 |
|---|---:|---:|---:|---:|
| Branch 1 | `0.00` | `0.00` | `0.00` | `0.00` |
| Branch 2 | `-3.53` | `-2.27` | `-2.67` | `-2.86` |
| Branch 3 | `-1.14` | `-0.77` | `-1.05` | `-0.71` |
| Branch 4 | `-9.50` | `-7.96` | `-9.37` | `-7.78` |

Branch 1 remains fixed because it is a singleton. For PPO Dirichlet, branches 2, 3, and 4 all become less sharp in v2.3. For RCPO Dirichlet, branches 3 and 4 become clearly less sharp, while branch 2 does not improve in the same way.

The concentration bounds therefore have a measurable effect, but not an identical effect on every branch. Branch 4 still has the most negative entropy because it is the six-asset full-universe branch; its entropy should be compared with branch 4 from the earlier version, not directly with lower-dimensional branches.

The validation evidence is still mixed, but in a different way from the earlier snapshot:

- PPO Dirichlet now keeps much more of its early best policy than before.
- RCPO Dirichlet reaches a stronger best checkpoint, but drifts farther away from it later.

This means v2.3 reduces excessive sharpness, but does not fully solve Dirichlet policy drift. It improves the PPO case more clearly than the RCPO case.

**Figure Placeholder: Dirichlet entropy by branch, v2.2 versus v2.3**

Suggested content:

- One panel for PPO Dirichlet and one for RCPO Dirichlet.
- Separate trajectories for branches 1-4.
- The same y-axis within each branch comparison.

**Figure Placeholder: Dirichlet validation stability**

Source:

- `evaluation/section9_simplex_v2.3_policy_comparison/section9_validation_score_history.png`


---

## 12. Shared 20-Market Best-Checkpoint Evaluation

**Key Words**

- Same future continuation markets
- Clean deterministic evaluation
- Best checkpoints
- Constrained-neutral baseline
- Hard versus soft allocation feasibility

**Evidence**

All five policies were evaluated on the same 20 future continuation markets. The constrained-neutral CAOSD allocation is the shared baseline.

| Policy | Mean Future Return | Excess vs Baseline | Win Rate | Mean Max Drawdown | Mean Turnover | Allocation Violation |
|---|---:|---:|---:|---:|---:|---:|
| PPO Gaussian v2.3 | `18.21%` | `+2.89%` | `60%` | `13.89%` | `13.13%` | `0` |
| PPO Dirichlet v2.3 | `18.74%` | `+3.42%` | `80%` | `14.33%` | `0.31%` | approximately `0` |
| RCPO Gaussian v2.3 | `16.66%` | `+1.34%` | `65%` | `14.07%` | `27.73%` | `0` |
| RCPO Dirichlet v2.3 | `18.15%` | `+2.83%` | `65%` | `13.74%` | `9.08%` | approximately `0` |
| Pure RCPO allocation penalty | `17.13%` | `+1.81%` | `60%` | `17.53%` | `67.00%` | nonzero |
| Constrained-neutral baseline | `15.32%` | `0` | N/A | `13.88%` | `0%` | `0` |

**Figure Placeholder: Mean cumulative return on 20 future markets**

Source:

- `evaluation/section9_simplex_v2.3_policy_comparison/section9_cumulative_return_means_only.png`

**Interpretation**

The best v2.3 checkpoints still all beat the constrained-neutral baseline in mean return.

The pure RCPO allocation-penalty policy also beats the baseline, but it has:

- Lower mean excess return than three of the four simplex policies.
- Higher maximum drawdown.
- Much higher turnover.
- Nonzero allocation violations on future markets.

Its future average raw allocation-violation cost is approximately `0.0356`, corresponding to scaled cost `0.00178`, above the training target `alpha = 0.0005`.

One new nuance is that the current shared-future evaluation no longer ranks the policies in the same order as their validation history. `RCPO Gaussian v2.3` still looks strong on validation, but on the shared 20-market test it now has the weakest mean excess return among the simplex policies. This makes the generalization question more concrete: better rollout control or better validation can still fail to translate into the strongest shared-future result.

**Figure Placeholder: Maximum drawdown comparison**

Source:

- `evaluation/section9_simplex_v2.3_policy_comparison/section9_max_drawdown_comparison.png`

**Figure Placeholder: Turnover comparison**

Source:

- `evaluation/section9_simplex_v2.3_policy_comparison/section9_turnover_comparison.png`

This is useful evidence for Simplex Decomposition:

```text
RCPO can learn to reduce allocation violations,
but CAOSD guarantees feasibility and generalizes that feasibility automatically.
```

---

## 13. Current Conclusions

**Key Words**

- Soft RCPO is informative but not hard-feasible
- Gaussian optimization is healthier
- Dirichlet sharpness is reduced
- Dirichlet drift remains
- Best checkpoint selection matters

**Explanation**

1. The pure RCPO allocation baseline learns to reduce constraint cost on training rollouts, but it still does not preserve feasibility as reliably as Simplex Decomposition on future markets.

2. The v2.3 Gaussian changes substantially improve optimization efficiency compared with v2.2, even though the KL early-stop rate rises again late in training.

3. The tighter Dirichlet concentration range still makes the high-dimensional branches less sharp than in v2.2, but reduced sharpness alone does not guarantee stable validation or shared-future performance.

4. The stability story is now asymmetric. PPO Dirichlet retains more of its early best checkpoint than before, while RCPO Dirichlet drifts more than the earlier 18k-update snapshot suggested.

5. The current shared 20-market evaluation favors `PPO Dirichlet v2.3`, followed by `PPO Gaussian v2.3` and `RCPO Dirichlet v2.3`. `RCPO Gaussian v2.3` remains hard-feasible, but its current shared-future mean excess is weaker than both PPO simplex policies and the RCPO Dirichlet simplex policy.

6. The most defensible current project structure is:

```text
Simplex Decomposition -> hard allocation feasibility
RCPO                    -> adaptive control of an additional dynamic risk
Pure allocation RCPO    -> comparison baseline showing the limits of soft feasibility
```

---

## 14. Discussion Questions

1. Should the pure RCPO allocation method be presented as a competing method or primarily as an ablation demonstrating the value of hard feasibility?

2. Is the large reduction in Gaussian KL early stopping enough to keep the current optimizer settings, or should the runs also be compared by completed optimizer steps?

3. For Dirichlet, should the next intervention regularize total concentration directly, schedule entropy, or use validation-based early stopping?

4. Is turnover important enough to become another formal constraint, particularly for the soft RCPO baseline?

5. After completing v2.3, should the next priority be repeated seeds or transition to real-market data?

---

## 15. One-Minute Summary

Since the previous meeting, I added a pure RCPO baseline that learns the two allocation constraints as a soft penalty without Simplex Decomposition. It reduces violations during training, but on 20 future markets it still has nonzero violations, very high turnover, and weaker overall behavior than the hard-simplex policies. This continues to support the main motivation for Simplex Decomposition: feasibility is guaranteed structurally rather than learned approximately.

Longer v2.2 training exposed two different policy problems. Gaussian runs triggered KL early stopping in about 94 to 96 percent of updates, while Dirichlet runs completed nearly all updates but could become sharp and lose validation performance. In v2.3, I halved the learning rate, tightened PPO clipping, and bounded Dirichlet concentration between 0.5 and 8. Gaussian early stopping is still far lower than in v2.2, but after very long training it has climbed back to about 11 to 15 percent overall and roughly 18 to 22 percent over the most recent 1,000 updates. Dirichlet entropy remains less sharp than in v2.2, especially in the large branch, but stability is still mixed: PPO Dirichlet currently preserves more of its best checkpoint, while RCPO Dirichlet drifts later. On the shared 20-market comparison, `PPO Dirichlet v2.3` currently gives the strongest mean excess return, followed by `PPO Gaussian v2.3`, while `RCPO Gaussian v2.3` is now the weakest of the simplex policies on this shared-future test.

---

## Run References

**v2.3 Simplex Runs**

- `runs/simplex_v2.3_ppo_unconstrained_none_20260707_204433/seed_0`
- `runs/simplex_v2.3_ppo_unconstrained_none_20260707_204606/seed_0`
- `runs/simplex_v2.3_rcpo_none_20260707_204441/seed_0`
- `runs/simplex_v2.3_rcpo_none_20260707_204616/seed_0`

**Pure RCPO Allocation-Penalty Baseline**

- `runs/rcpo_allocation_penalty_rcpo_none_20260630_141127/seed_0`

**Shared Evaluation**

- `evaluation/section9_simplex_v2.3_policy_comparison`
