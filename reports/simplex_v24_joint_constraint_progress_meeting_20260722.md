# Constraint-Aware Portfolio RL: v2.4 and Joint-Penalty Progress

**Meeting date:** 22 July 2026  
**Project:** Constraint-Aware Simplex Decomposition for Robust Portfolio Reinforcement Learning  
**Status:** Shared meeting handout; v2.4 results use the best checkpoints available on 22 July 2026

---

## 1. Meeting Goal And Progress Since The Previous Meeting

### Key Words

`four solutions` | `longer v2.3 training` | `policy-specific v2.4 tuning` | `joint soft penalty` | `return-risk-feasibility`

### Meeting Goal

This meeting evaluates two developments since the previous progress report:

1. A fourth constraint solution has been implemented: pure RCPO learns both allocation and maximum-drawdown constraints through a joint soft penalty, without Simplex Decomposition.
2. The four Simplex policies were trained substantially longer in v2.3, diagnosed, and retuned separately in v2.4 rather than sharing one optimizer configuration. RCPO now saves both a return-best checkpoint and a feasible-best checkpoint.

The main question is no longer simply which method has the highest return:

> Which method produces return while remaining feasible, stable, sufficiently adaptive, and not excessively concentrated in one asset?

---

## 2. The Four Constraint Solutions

### Key Words

`soft feasibility` | `hard feasibility` | `drawdown control` | `combined penalty`

| Solution | Policy / action mechanism | Active RCPO cost | Allocation feasibility |
|---|---|---|---|
| 1. RCPO allocation penalty | Flat Gaussian policy with ordinary softmax | Allocation violation | Learned softly; violations are possible |
| 2. Pure Simplex Decomposition | CAOSD action mapping with PPO | None | Guaranteed by construction |
| 3. Simplex + RCPO | CAOSD action mapping with RCPO | Benchmark-relative maximum drawdown | Allocation is hard; drawdown is learned softly |
| 4. RCPO allocation + drawdown penalty | Flat Gaussian policy with ordinary softmax | Combined allocation and drawdown cost | Both constraints are learned softly |


---

## 3. New Fourth Solution: Joint Allocation And Drawdown RCPO

### Key Words

`no simplex` | `one lambda` | `allocation cost` | `drawdown cost` | `soft constraints`

### Constraint Definition

The new policy uses an ordinary softmax portfolio action. Its RCPO cost combines two terms:

```text
combined_constraint_cost_t
    = allocation_constraint_cost_t
    + 0.25 * drawdown_constraint_cost_t
```

The allocation component measures violation of the two minimum group-weight requirements. The drawdown component measures excess maximum drawdown relative to the online constrained-neutral benchmark budget.


- Drawdown cost is weighted by `0.25`, preventing the second constraint from overwhelming the first.


---

## 5. Configuration Differences: v2.3 Versus v2.4

### What Stayed The Same

Both versions use the same synthetic market, eight training markets, 252-step episodes, CAOSD allocation constraints, standalone branch credit, minibatch size `512`, PPO clip `0.10`, and target KL `0.02`. Therefore, the main v2.3-to-v2.4 differences are optimizer scheduling, Dirichlet concentration settings, and constraint strictness.


### v2.4: Policy-Specific Active Settings

The table below uses the active PPO block for PPO runs and the active optimization block for RCPO runs recorded in each run snapshot.

| Policy | v2.3 learning rate | v2.4 learning-rate schedule | Epochs: v2.3 -> v2.4 | Additional v2.4 change | Intended effect |
|---|---:|---:|---:|---|---|
| PPO Gaussian | `1.0e-4` constant | `9.5e-5 -> 5.0e-5` | `4 -> 3` | No distribution change | Reduce KL overshoot and late policy drift |
| PPO Dirichlet | `1.0e-4` constant | `9.0e-5 -> 4.0e-5` | `4 -> 4` | Entropy `0.0015`; concentration `1.0 / 2.0 / 12.0` | Preserve exploration while avoiding early boundary collapse |
| RCPO Gaussian | `1.0e-4` constant | `7.0e-5 -> 3.0e-5` | `4 -> 3` | `lambda_lr_up: 0.0015 -> 0.00075` | Reduce both KL overshoot and rapid constraint dominance |
| RCPO Dirichlet | `1.0e-4` constant | `7.5e-5 -> 3.5e-5` | `4 -> 4` | Concentration remains `0.5 / 1.5 / 8.0` | Stabilize learning without removing useful stochasticity |

### Constraint And Evaluation Changes

| Setting | v2.3 | v2.4 | Meaning |
|---|---:|---:|---|
| Benchmark drawdown margin | `0.95` | `0.90` | v2.4 asks for approximately 10% lower drawdown than the benchmark |
| Dynamic alpha budget ratio | `0.05` | `0.04` | v2.4 tolerates a smaller relative violation |




---

## 6. Training And Validation Stability

### Key Words

`KL early stop` | `completed optimizer steps` | `best versus latest` | `checkpoint retention`

### Optimizer Stability

| Policy | Version | Overall KL early-stop rate | Recent 1,000 updates | Completed optimizer steps |
|---|---|---:|---:|---:|
| PPO Gaussian | v2.3 | 38.7% | 56.6% | 13.80 / 16 |
| PPO Gaussian | v2.4 | **3.5%** | **9.5%** | 11.83 / 12 |
| RCPO Gaussian | v2.3 | 52.2% | 81.4% | 12.05 / 16 |
| RCPO Gaussian | v2.4 | **6.2%** | **8.0%** | 11.73 / 12 |
| PPO Dirichlet | v2.3 | 0.01% | 0.0% | 16.00 / 16 |
| PPO Dirichlet | v2.4 | 0.01% | 0.0% | 16.00 / 16 |
| RCPO Dirichlet | v2.3 | 0.11% | 0.0% | 16.00 / 16 |
| RCPO Dirichlet | v2.4 | 0.13% | 0.0% | 16.00 / 16 |

The Gaussian retuning clearly solved the immediate KL problem. Recent RCPO Gaussian early stopping fell from approximately 81% to 8%. Dirichlet was already KL-stable; its remaining issue is economic policy quality rather than interrupted optimization.

### Validation Learning And Retention

| Policy | Version | Best validation excess | Latest validation excess | Drop from best |
|---|---|---:|---:|---:|
| PPO Gaussian | v2.3 | +12.46% | +5.28% | 7.18 pp |
| PPO Gaussian | v2.4 | +11.61% | +7.72% | 3.89 pp |
| PPO Dirichlet | v2.3 | +9.26% | +5.86% | 3.40 pp |
| PPO Dirichlet | v2.4 | **+13.76%** | **+12.27%** | **1.49 pp** |
| RCPO Gaussian | v2.3 | +6.65% | +5.30% | 1.35 pp |
| RCPO Gaussian | v2.4 | **+12.08%** | **+10.71%** | 1.37 pp |
| RCPO Dirichlet | v2.3 | **+10.31%** | +7.16% | 3.15 pp |
| RCPO Dirichlet | v2.4 | +9.23% | **+9.08%** | **0.15 pp** |

### Main Findings

- PPO Dirichlet is the clearest v2.4 training improvement: both its peak validation score and post-peak retention improved.
- RCPO Gaussian learned a much stronger validation policy after KL retuning.
- RCPO Dirichlet has a slightly lower peak than v2.3 but is exceptionally stable.
- PPO Gaussian is more stable than v2.3 but does not yet show a stronger best policy.
- KL stability is necessary but not sufficient: RCPO Gaussian later shows unhealthy cost-advantage dominance despite its controlled KL.

[Figure Placeholder: v2.3 and v2.4 validation score histories]

Suggested sources:

- `evaluation/section9_simplex_v2.3_policy_comparison/section9_validation_score_history.png`
- `evaluation/section9_simplex_v2.4_policy_comparison/section9_validation_score_history.png`

---

## 7. Shared 20-Market Evaluation: v2.3 Versus v2.4

### Evaluation Protocol

All methods use their current return-best checkpoint and are evaluated deterministically on the same 20 future continuation markets. Evaluation uses clean realized returns. The constrained-neutral CAOSD portfolio is the common baseline.

### v2.3 Results

| Policy | Mean return | Excess vs baseline | Win rate | Mean max drawdown | Mean turnover |
|---|---:|---:|---:|---:|---:|
| PPO Gaussian v2.3 | 18.21% | +2.88% | 65% | 15.56% | 26.20% |
| PPO Dirichlet v2.3 | 18.18% | +2.86% | 70% | 14.29% | 27.19% |
| RCPO Gaussian v2.3 | 16.66% | +1.34% | 65% | 14.07% | 27.73% |
| RCPO Dirichlet v2.3 | **19.10%** | **+3.78%** | 65% | **13.18%** | 27.03% |
| Constrained-neutral baseline | 15.32% | 0.00% | - | 13.88% | 0.00% |

RCPO Dirichlet gave the strongest v2.3 balance: the highest mean return and the lowest mean maximum drawdown among the learned policies.

### v2.4 And Soft-RCPO Results

| Policy | Mean return | Excess vs baseline | Win rate | Mean max drawdown | Mean turnover |
|---|---:|---:|---:|---:|---:|
| PPO Gaussian v2.4 | 16.05% | +0.73% | 60% | 15.05% | 27.00% |
| PPO Dirichlet v2.4 | 17.52% | +2.20% | 55% | 14.57% | 23.73% |
| RCPO Gaussian v2.4 | 19.65% | +4.33% | 70% | 14.85% | **1.10%** |
| RCPO Dirichlet v2.4 | 18.43% | +3.10% | **80%** | **13.91%** | 9.72% |
| RCPO allocation-only v2 | **20.42%** | **+5.10%** | 70% | 16.71% | 6.17% |
| RCPO allocation + drawdown | 19.04% | +3.72% | 70% | 15.58% | 9.08% |
| Constrained-neutral baseline | 15.32% | 0.00% | - | 13.88% | 0.00% |

### Cross-Version Interpretation

- Allocation-only RCPO has the highest return, but also the highest maximum drawdown.
- RCPO Dirichlet v2.4 has the best win rate and drawdown closest to the baseline.
- RCPO Gaussian v2.4 has high return and extremely low turnover, but weight diagnostics show that it learned an almost static portfolio.
- Adding drawdown cost to allocation RCPO reduces concentration and drawdown relative to allocation-only RCPO, but its return-best checkpoint remains over the joint cost target.
- PPO Gaussian remains the weakest v2.4 policy by mean future excess return.

[Figure Placeholder: v2.3 versus v2.4 cumulative-return comparison]

[Figure Placeholder: v2.4 maximum-drawdown and turnover comparison]

Suggested sources:

- `evaluation/section9_simplex_v2.3_policy_comparison/section9_cumulative_return_comparison.png`
- `evaluation/section9_simplex_v2.4_policy_comparison/section9_cumulative_return_comparison.png`
- `evaluation/section9_simplex_v2.4_policy_comparison/section9_max_drawdown_comparison.png`
- `evaluation/section9_simplex_v2.4_policy_comparison/section9_turnover_comparison.png`

---

## 8. Direct Interpretation Of v2.3 -> v2.4

### PPO Gaussian

- KL stability improved substantially.
- Future excess return decreased from `+2.88%` to `+0.73%`.
- Drawdown improved slightly, while turnover remained high.
- Conclusion: the optimizer is healthier, but the learned strategy is not yet better. Removing early stopping was necessary but not sufficient.

### PPO Dirichlet

- Validation best and validation retention improved strongly.
- Shared-test excess return decreased moderately from `+2.86%` to `+2.20%`.
- Turnover declined from `27.19%` to `23.73%`.
- Conclusion: training is more stable, but the validation improvement has not fully transferred to the shared future markets.

### RCPO Gaussian

- KL stability and mean return improved strongly.
- Shared-test excess increased from `+1.34%` to `+4.33%`.
- Turnover collapsed from `27.73%` to `1.10%`.
- Conclusion: this is not simply a better dynamic policy. It has converged to a nearly fixed, constraint-boundary allocation. Its economic interpretation must be separated from its return score.

### RCPO Dirichlet

- Shared-test excess decreased from `+3.78%` to `+3.10%`.
- Win rate improved to `80%`, and turnover fell to `9.72%`.
- Mean maximum drawdown remains close to the constrained-neutral baseline.
- Conclusion: v2.4 sacrifices some peak return but produces a more stable and credible constrained strategy.

---

## 9. Allocation-Weight And Concentration Diagnostics

### Key Words

`asset 5 shortcut` | `sum(weights^2)` | `maximum exposure` | `static policy` | `dynamic adaptation`

The two allocation groups overlap on Asset 5. A policy can satisfy both minimum-weight constraints efficiently by assigning a large weight to this one asset. Therefore, allocation feasibility alone can hide a concentrated shortcut.

### Concentration Definition

```text
concentration_t = sum_i(weight_i,t^2)
```

Lower values indicate greater diversification. Arithmetic equal weight over six assets would be `1/6 = 0.167`, while the constrained-neutral CAOSD baseline has concentration `0.274` because the hard allocation requirements prevent ordinary equal weight.

### Shared 20-Market Concentration Results

| Policy | Version | Mean concentration | Mean largest asset weight | Worst asset weight | Steps above 70% in one asset |
|---|---|---:|---:|---:|---:|
| PPO Gaussian | v2.3 | 0.454 | 57.0% | 99.6% | 23.4% |
| PPO Gaussian | v2.4 | **0.417** | **51.4%** | 99.6% | **19.4%** |
| PPO Dirichlet | v2.3 | **0.423** | **51.7%** | 85.2% | **22.7%** |
| PPO Dirichlet | v2.4 | 0.446 | 56.2% | **81.0%** | 36.1% |
| RCPO Gaussian | v2.3 | **0.339** | **42.5%** | 70.3% | 0.0% |
| RCPO Gaussian | v2.4 | 0.429 | 54.3% | **54.8%** | 0.0% |
| RCPO Dirichlet | v2.3 | 0.353 | **44.9%** | 84.8% | 5.2% |
| RCPO Dirichlet | v2.4 | 0.361 | 49.7% | **68.9%** | **0.0%** |
| RCPO allocation-only v2 | - | 0.649 | 79.3% | 93.3% | 97.7% |
| RCPO allocation + drawdown | - | 0.511 | 69.7% | 85.6% | 46.6% |
| Constrained-neutral baseline | - | **0.274** | **42.5%** | **42.5%** | 0.0% |

### Strategy Interpretation

- **PPO Gaussian:** average concentration improved in v2.4, but the 99.6% worst-case exposure remains. It still occasionally makes nearly all-in allocations.
- **PPO Dirichlet:** it avoids the Gaussian policy's most extreme worst-case spike, but v2.4 is more concentrated on average and spends 36.1% of steps above 70% in one asset.
- **RCPO Gaussian:** its largest weight is tightly bounded near 54%, but combined with turnover of only 1.10%, this indicates a nearly fixed portfolio rather than strong state adaptation.
- **RCPO Dirichlet:** it has the most credible v2.4 Simplex balance: concentration `0.361`, no steps above 70%, and nonzero but moderate turnover.
- **Allocation-only RCPO:** it exploits the overlap directly, holding more than 70% in one asset for 97.7% of evaluation steps.
- **Joint allocation-drawdown RCPO:** drawdown pressure reduces concentration from `0.649` to `0.511`, but the strategy remains strongly Asset-5 dominated.

### Main Finding

High return and zero allocation violation do not necessarily mean a policy learned a robust allocation rule. The allocation-only RCPO and Gaussian Simplex RCPO illustrate two different shortcuts: extreme overlap-asset concentration and a favorable static portfolio.

[Figure: Shared 20-market portfolio concentration]

Source:

- `evaluation/section9_simplex_v2.4_policy_comparison/section9_concentration_comparison.png`

[Figure Placeholder: Group and asset allocation paths]

Suggested sources:

- `runs/simplex_v2.4_rcpo_gaussian_rcpo_none_20260718_162411/seed_0/evaluation/group_weights_validation.png`
- `runs/simplex_v2.4_rcpo_dirichlet_rcpo_none_20260718_162422/seed_0/evaluation/group_weights_validation.png`
- `runs/rcpo_allocation_penalty_v2_rcpo_none_20260718_162428/seed_0/evaluation/group_weights_validation.png`
- `runs/rcpo_allocation_drawdown_penalty_rcpo_none_20260718_162434/seed_0/evaluation/group_weights_validation.png`

---

## 10. Constraint Health Of The RCPO Policies

### Key Words

`lambda scale` | `advantage dominance` | `cost/alpha` | `feasible checkpoint`

| RCPO policy | Latest lambda | Constraint-cost status | Policy-pressure diagnosis |
|---|---:|---|---|
| Simplex Gaussian v2.4 | 1.06 | Cost approximately 11.3 times alpha | `lambda * A_cost` about 5.3 times reward-advantage scale; unhealthy constraint dominance |
| Simplex Dirichlet v2.4 | 0.93 | Cost approximately 6.2 times alpha | Penalty/reward advantage ratio about 0.73; strong but still manageable |
| Allocation-only RCPO v2 | 0.00 | Latest cost below alpha | Feasible at latest and best checkpoints, but highly concentrated |
| Allocation + drawdown RCPO | 0.21 | Latest cost approximately 8.6 times alpha | Penalty/reward advantage ratio about 0.34; learning remains return-sensitive, but return-best is infeasible |

### Interpretation

The v2.4 Gaussian RCPO has healthy KL behavior but unhealthy reward-cost balance. Its very low turnover is consistent with a large lambda discouraging movement away from a safe static allocation.

This is an important distinction:

> KL stability means the optimizer is taking controlled policy steps. It does not mean the RCPO objective is balanced.

The Dirichlet RCPO currently has the most credible balance among the Simplex RCPO policies. The joint soft-penalty RCPO also remains return-sensitive, but it needs feasible-checkpoint evaluation before it can be called successful under both constraints.

---

## 11. Gaussian Versus Dirichlet: Pareto And Feasibility Evidence

### How To Read The Pareto Plot

The new plot uses:

- x-axis: mean maximum drawdown, where lower is better;
- y-axis: mean excess cumulative return, where higher is better;
- marker size: mean portfolio concentration;
- filled marker: the method meets its active constraint target;
- hollow marker: the method violates its active constraint target.

For PPO Simplex, the active constraint is hard allocation feasibility. For Simplex RCPO, it also includes the dynamic drawdown target. For allocation-only and joint-penalty RCPO, it includes their fixed-alpha soft cost target.

### v2.4 Feasibility Results

| Policy | Mean excess return | Mean max drawdown | Mean concentration | Feasible future branches | Overall active-cost feasible? |
|---|---:|---:|---:|---:|---|
| PPO Gaussian | +0.73% | 15.05% | 0.417 | 100% | Yes: hard allocation only |
| PPO Dirichlet | +2.20% | 14.57% | 0.446 | 100% | Yes: hard allocation only |
| RCPO Gaussian | +4.33% | 14.85% | 0.429 | 15% | **No** |
| RCPO Dirichlet | +3.10% | 13.91% | 0.361 | 20% | **No** |
| RCPO allocation-only | +5.10% | 16.71% | 0.649 | 100% | Yes, but highly concentrated |
| RCPO allocation + drawdown | +3.72% | 15.58% | 0.511 | 30% | **No** |
| Constrained-neutral baseline | 0.00% | 13.88% | 0.274 | 100% | Yes |

### Is Dirichlet More Promising?

The answer depends on the objective.

- **For PPO:** Dirichlet has higher return and lower drawdown, and avoids Gaussian's 99.6% worst-case allocation. However, it is more concentrated on average and spends more time above 70% in one asset. It is not uniformly superior.
- **For RCPO:** Gaussian has higher return, but its portfolio is nearly static and its cost advantage dominates the reward advantage. Dirichlet has lower drawdown, lower concentration, an 80% win rate, and more credible adaptation. It is currently the stronger constrained-control candidate.
- **For strict feasibility:** neither return-best Simplex RCPO checkpoint is satisfactory. Feasible-best checkpoint evaluation is required before selecting a final RCPO architecture.

### Balanced Conclusion

Dirichlet is more promising for adaptive RCPO under multiple evaluation criteria, but Gaussian remains useful as a simpler and sometimes higher-return comparator. The Pareto evidence argues against selecting either distribution from return alone.

[Figure: Return-risk-feasibility Pareto comparison]

Source:

- `evaluation/section9_simplex_v2.4_policy_comparison/section9_return_risk_feasibility_pareto.png`

---

## 12. Future Research Path: Multi-Constraint Simplex RCPO

### Motivation From The Joint Soft-Penalty Baseline

The joint allocation-drawdown RCPO shows that one policy can learn from more than one cost. However, combining different costs before the lambda update hides which risk is driving constraint pressure. The same idea can be extended more cleanly inside Simplex RCPO by keeping hard allocation feasibility and assigning separate multipliers to different soft, path-dependent risks.

### Proposed Policy Advantage

```text
A_policy
    = A_return
    - lambda_drawdown * A_drawdown
    - lambda_turnover * A_turnover
    - lambda_tail * A_tail_loss
```

Possible cost roles:

- `A_drawdown`: discourage peak-to-trough portfolio loss beyond a benchmark-relative budget.
- `A_turnover`: discourage excessive reallocation and transaction-cost-sensitive behavior.
- `A_tail_loss`: discourage severe lower-tail losses, for example through a CVaR-like or threshold-loss cost.

Each cost would have its own critic, tolerance, scale, and multiplier update:

```text
lambda_i <- max(0, lambda_i + learning_rate_i * (observed_cost_i - alpha_i))
```

### Why Separate Lambdas May Help

- Each multiplier has a clear economic interpretation.
- A large turnover violation cannot be hidden by a small drawdown cost, or vice versa.
- The learned trade-off can be inspected through `lambda_i * A_cost_i / A_return`.
- Simplex Decomposition continues guaranteeing allocation feasibility, so lambdas are reserved for dynamic risks that are difficult to encode as a fixed action region.

### Main Risk

Multiple multipliers can jointly dominate return learning. The implementation should therefore be staged:

1. Decompose and log the existing joint cost without changing optimization.
2. Add a second multiplier for turnover while keeping the current drawdown cost.
3. Add tail-loss control only after reward-versus-cost scale is healthy.
4. Require feasible-best checkpoint selection across every active constraint.

This path turns the current one-lambda experiment into a more interpretable multi-objective constrained RL extension rather than simply adding a larger combined penalty.


---

## Run And Artifact References

### v2.3 Simplex Runs

- Gaussian PPO: `runs/simplex_v2.3_ppo_unconstrained_none_20260707_204433/seed_0`
- Dirichlet PPO: `runs/simplex_v2.3_ppo_unconstrained_none_20260707_204606/seed_0`
- Gaussian RCPO: `runs/simplex_v2.3_rcpo_none_20260707_204441/seed_0`
- Dirichlet RCPO: `runs/simplex_v2.3_rcpo_none_20260707_204616/seed_0`

### v2.4 Simplex Runs

- Gaussian PPO: `runs/simplex_v2.4_ppo_gaussian_ppo_unconstrained_none_20260718_162406/seed_0`
- Gaussian RCPO: `runs/simplex_v2.4_rcpo_gaussian_rcpo_none_20260718_162411/seed_0`
- Dirichlet PPO: `runs/simplex_v2.4_ppo_dirichlet_ppo_unconstrained_none_20260718_162418/seed_0`
- Dirichlet RCPO: `runs/simplex_v2.4_rcpo_dirichlet_rcpo_none_20260718_162422/seed_0`

### Soft RCPO Runs

- Allocation-only v2: `runs/rcpo_allocation_penalty_v2_rcpo_none_20260718_162428/seed_0`
- Allocation + drawdown: `runs/rcpo_allocation_drawdown_penalty_rcpo_none_20260718_162434/seed_0`

### Shared Evaluations

- v2.3: `evaluation/section9_simplex_v2.3_policy_comparison`
- v2.4 and soft RCPO baselines: `evaluation/section9_simplex_v2.4_policy_comparison`
