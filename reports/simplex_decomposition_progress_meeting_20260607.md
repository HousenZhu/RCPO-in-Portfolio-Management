

## 1. Project Direction


- Hard feasibility: allocation constraints
- Soft adaptive control: drawdown risk


The project now separates two kinds of constraints:

- Hard allocation constraints are handled through Simplex Decomposition.
- Path-dependent risk constraints, especially maximum drawdown, are handled through an RCPO-inspired adaptive control layer.


Simplex Decomposition says the action is always feasible. RCPO says among feasible actions, avoid policies that create too much drawdown."

---

## 2. Brief Review Of The Simplex Decomposition Paper

**Key Words**

- CAOSD: constraint-aware ordered simplex decomposition
- Two allocation constraints
- Four branch simplexes
- `V1 ∩ V2`, `V1`, `V2`, full universe
- Recombine with `z1..z4`
- Autoregressive Dirichlet policy in the paper

**Explanation**

 For two constraints, the method decomposes the constrained action region into four branch simplexes:

- Branch 1: assets in both constrained sets, `V1 ∩ V2`
- Branch 2: assets in the first constrained set, `V1`
- Branch 3: assets in the second constrained set, `V2`
- Branch 4: the full asset universe

Each branch produces a valid simplex allocation over its own asset subset. The final portfolio is a weighted sum of the padded branch allocations:

```text
w = z1 y1 + z2 y2 + z3 y3 + z4 y4
```


The paper combines this decomposition with autoregressive Dirichlet branch policies, transformer-style market encoding, PPO training, real Nasdaq data, and random constraint configurations.



---

## 3. My Current Algorithm Compared With The Paper

**Key Words**

- Preserved: CAOSD hard feasibility mapping
- Changed: Gaussian logits instead of Dirichlet weights
- Added: RCPO maximum-drawdown control
- Added: constrained benchmark

**Explanation**

The current implementation keeps the core CAOSD feasibility mechanism, but it is not a full reproduction of the paper.

Current implementation:

- Builds the same four branch sets: `V1 ∩ V2`, `V1`, `V2`, and all assets.
- Converts branch logits into branch allocations through softmax.
- Recombines branch allocations using the CAOSD `z1..z4` construction.
- Guarantees that the final portfolio satisfies the configured allocation constraints.

Important differences from the paper:

- The paper uses autoregressive Dirichlet branch policies; the current code uses Gaussian branch logits followed by softmax.
- The paper uses a transformer-style market encoder; the current code uses an MLP actor-critic.
- The current project adds an RCPO-inspired maximum-drawdown control layer on top of hard allocation feasibility.


---

## 4. Experiment Stage 1: Flat Combined Branch Output

**Key Words**

- One flat output vector
- All branch logits in one head
- Weak learning signal
- Abandoned quickly

**Explanation**

The first practical version used one combined actor output for all branch logits. 

it did not show a useful learning signal. 


---

## 5. Experiment Stage 2: Parallel Simplex Branch Gaussian

**Key Words**

- Shared encoder
- Four parallel Gaussian branch heads
- Hard feasible CAOSD weights
- PPO: no clear success
- RCPO: unstable, high lambda

**Explanation**

The second version used a shared neural encoder with four parallel branch actor heads. Each head produced Gaussian logits for one CAOSD branch. The branch logits were softmaxed inside the environment and recombined into feasible portfolio weights.

 the experiments still did not show a strong successful signal. PPO remained weak, and RCPO became unstable in the branch Gaussian run.

Evidence from completed runs:

| Run | Policy | Latest Validation Excess | Best Validation Excess | Lambda | Observation |
|---|---:|---:|---:|---:|---|
| `simplex_v1_ppo_unconstrained_none_20260522_231007` | Branch Gaussian PPO | `-0.0080` | `+0.0133` | `0.000` | Weak validation signal |
| `simplex_v1_rcpo_none_20260602_162102` | Branch Gaussian RCPO | `-0.0197` | `+0.0021` | `11.275` | High lambda and negative validation excess |



This told me that just splitting the output into branches is not enough. The branches are not independent in the final portfolio because the CAOSD recombination links them. That motivated the third version, where later branches condition on previous branch allocations."

---

## 6. Experiment Stage 3: Autoregressive Simplex Gaussian

**Key Words**

- Shared encoder
- Later heads condition on previous branch allocations
- Gaussian logits, not Dirichlet
- RCPO works after rescaling
- PPO alone still weak

**Explanation**

The third version keeps Gaussian branch logits but makes the branch policy autoregressive. The first branch is generated from the shared state features. Later branches receive both the shared features and previous branch softmax allocations.



Evidence:

| Run | Policy | Latest Validation Excess | Best Validation Excess | Lambda | Observation |
|---|---:|---:|---:|---:|---|
| `simplex_v1_ppo_unconstrained_none_20260526_125507` | Autoregressive Gaussian PPO | `+0.0074` | `+0.0135` | `0.000` | Still weak / no clear success |
| `simplex_v1_rcpo_none_20260602_162145` | Autoregressive Gaussian RCPO, before cost rescale | `+0.0524` | `+0.0591` | `8.314` | Return improved, but lambda became too dominant |
| `simplex_v1_rcpo_none_20260606_150515` | Autoregressive Gaussian RCPO, adjusted cost scale | `+0.0391` | `+0.0402` | `0.241` | Current healthier run, in progress |


---

## 7. What I Learned About RCPO Lambda And Constraint Cost

**Key Words**

- Initial misunderstanding: lambda looked like a small scalar
- Real effect: `lambda * A_cost`
- Cost advantage can dominate reward advantage
- Drawdown cost scale changed 

**Explanation**

RCPO uses a combined policy advantage:

```text
A_combined = A_reward - lambda * A_cost
```

This means the real strength of the constraint depends on both lambda and the scale of the cost advantage. A lambda value that looks moderate can dominate training if `A_cost` has much larger variance than `A_reward`.

When lambda is 8.314 :

```text
scale(lambda * A_cost) is 140X compared with scale(A_reward)
```

---

## 8. Current Result After Rescaling The Constraint


Current evidence from the latest metrics:

| Metric | Value |
|---|---:|
| Latest validation excess | `+0.0391` |
| Best validation excess | `+0.0402` |
| Latest validation constraint cost | `0.00065` |
| Latest validation alpha | `0.00037` |
| Latest lambda | `0.241` |
| Maximum lambda observed | `0.241` |

Fresh rollout diagnostic from `checkpoint_last.pt`:

| Advantage term | Std | Mean Absolute Value | Interpretation |
|---|---:|---:|---|
| `A_return` | `0.03528` | `0.02675` | Main profit-seeking signal |
| `A_cost` | `0.02936` | `0.01744` | Drawdown-risk signal before lambda |
| `lambda * A_cost` | `0.00707` | `0.00420` | Effective constraint pressure |

The effective constraint term is now about `20.1%` of the return-advantage standard deviation and about `15.7%` of the return-advantage mean absolute value. 



---

## 9. Current Problem: Pure Simplex Autoregressive PPO Still Fails

**Key Words**

- Pure simplex PPO remains open
- Autoregressive structure alone is not enough
- RCPO may provide useful learning pressure
- Need understand whether issue is architecture, reward signal, exploration, or market design

**Explanation**

The current open problem is that the autoregressive simplex architecture does not clearly succeed with PPO alone. It produces some positive validation excess at the best checkpoint, but the signal is not strong enough to claim success.

This matters because ideally the simplex policy should learn a useful allocation rule even before adding RCPO. RCPO should improve risk behavior, not be the only reason the policy learns.

Possible explanations:

- The simplex action constraints make exploration harder.
- The Gaussian logit parameterization may still be less natural than the paper's Dirichlet branch policy.
- The synthetic market signal may not be strong enough under the constrained action space.
- PPO hyperparameters may need retuning for the simplex architecture.
- The model may need better branch-specific normalization or initialization.


---

## Evidence Summary

| Stage | Run | Architecture | Algorithm | Latest Validation Excess | Best Validation Excess | Latest Lambda | Main Interpretation |
|---|---|---|---|---:|---:|---:|---|
| 2 | `simplex_v1_ppo_unconstrained_none_20260522_231007` | Parallel branch Gaussian | PPO | `-0.0080` | `+0.0133` | `0.000` | No clear robust validation signal |
| 2 | `simplex_v1_rcpo_none_20260602_162102` | Parallel branch Gaussian | RCPO | `-0.0197` | `+0.0021` | `11.275` | Unstable under RCPO; lambda became very large |
| 3 | `simplex_v1_ppo_unconstrained_none_20260526_125507` | Autoregressive Gaussian | PPO | `+0.0074` | `+0.0135` | `0.000` | Slight positive best result, but not a convincing solution |
| 3 | `simplex_v1_rcpo_none_20260602_162145` | Autoregressive Gaussian | RCPO, before rescale | `+0.0524` | `+0.0591` | `8.314` | Return improved, but cost pressure became too dominant |
| 3 | `simplex_v1_rcpo_none_20260606_150515` | Autoregressive Gaussian | RCPO, adjusted cost scale | `+0.0391` | `+0.0402` | `0.241` | Healthier balance; current/in-progress evidence |

Notes:

- Validation excess means model final cumulative return minus the current baseline final cumulative return on validation branches.
- Latest adjusted RCPO evidence should be treated as current progress, not final experimental proof.
- Reward correction remains in the broader syllabus plan, but this meeting focuses on Simplex Decomposition and RCPO drawdown control.

---

## References

- Winkel et al., *Simplex Decomposition for Portfolio Allocation Constraints in Reinforcement Learning*. [arXiv](https://arxiv.org/abs/2404.10683), [DOI page](https://journals.sagepub.com/doi/10.3233/FAIA230573), [MCML entry](https://mcml.ai/publications/wss%2B23a/).
- Tessler, Mankowitz, and Mannor, *Reward Constrained Policy Optimization*, ICLR 2019. [OpenReview](https://openreview.net/forum?id=SkfrvsA9FX).
- Local syllabus: `HousenZhu_MEng_Project_Syllabus/main.tex`.
- Current implementation files: `src/rcpo_portfolio/simplex.py`, `src/rcpo_portfolio/models.py`, `src/rcpo_portfolio/env.py`.
