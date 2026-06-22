# Progress Meeting Notes: Branch Credit Assignment And Dirichlet Simplex Policy

## 1. Key Words



- Simplex Decomposition for hard allocation constraints
- RCPO for adaptive drawdown control
- New focus: branch-level credit assignment
- New focus: Dirichlet simplex policy

**Explanation**

The current project direction is to combine hard feasibility from Simplex Decomposition with adaptive risk control from RCPO. the main progress is on two technical problems:

- How to assign learning credit to the four CAOSD branches.
- How to implement and evaluate a Dirichlet policy for simplex-constrained branch allocations.

The current evidence suggests that Dirichlet can find a useful RCPO policy faster than Gaussian logits, but it also tends to lose performance later. Gaussian logits learn more slowly but eventually reach a slightly higher validation excess.

---

## 3. Brief Review: Simplex Decomposition Paper

**Key Words**

- CAOSD
- Two allocation constraints
- Four branch simplexes
- `V1 ∩ V2`, `V1`, `V2`, full universe
- Recombination by `z1, z2, z3, z4`

**Explanation**

The Simplex Decomposition paper decomposes a constrained simplex action space into smaller simplexes. For two allocation constraints, the final feasible action is built from four branch allocations:

```text
Branch 1: V1 ∩ V2
Branch 2: V1
Branch 3: V2
Branch 4: full asset universe
```

Each branch produces a valid simplex allocation on its own subset. The final portfolio is:

```text
w = z1 y1 + z2 y2 + z3 y3 + z4 y4
```

This guarantees that the final portfolio satisfies the allocation constraints.

---

## 4. Problem Found: Global Credit Assignment Was Weak

**Key Words**

- Four branches act together
- One portfolio-level reward
- Credit assignment problem
- Which branch caused return or drawdown?
- Global advantage was too coarse

**Explanation**

Initially, all branches received the same final portfolio-level advantage:

```text
A_global = A_return - lambda * A_cost
```

This is simple, but it creates a credit assignment problem. The final portfolio return is produced by all four branches together, so the algorithm does not directly know which branch helped or hurt.

This was especially problematic because CAOSD recombination gives different masses to different branches:

```text
w = z1 y1 + z2 y2 + z3 y3 + z4 y4
```

A branch with small `z_i` should not receive the same learning pressure as a branch with large `z_i`.

---

## 5. New Solution: Standalone Branch Return And Constraint Credit

**Key Words**

- Standalone branch path
- Same market returns
- Branch-specific return advantage
- Branch-specific cost advantage
- One global lambda

**Explanation**

I redesigned the credit assignment by giving each branch its own lightweight standalone portfolio path.

Each branch is not just an output vector; it is a sub-policy over a sub-simplex. Therefore, it should receive a learning signal related to its own allocation behavior.

For each branch, I compute:

```text
branch_return_i
branch_transaction_cost_i
branch_drawdown_i
branch_constraint_cost_i
```

Then each branch has its own advantage:

```text
A_i = A_reward_i - lambda * A_cost_i
```

The policy loss is weighted by the actual CAOSD mass:

```text
L_i = z_i * PPO_surrogate_i(A_i)
```

The final policy loss is:

```text
L_policy = sum_i L_i
```

**Important Design Choice**

There is still only one global lambda. Lambda is updated from the final combined portfolio drawdown cost, not from separate branch costs.

This means:

- Branches receive better local learning signals.
- The final portfolio remains the object of risk control.
- The method avoids creating four separate constraint controllers.

---

## 7. Dirichlet Policy Motivation

**Key Words**

- Inspired by Dirichlet power allocation
- Simplex-native distribution
- Direct allocation weights
- Alternative to Gaussian-softmax
- Bias and variance issue

**Explanation**

I was inspired by the paper **“A Prescriptive Dirichlet Power Allocation Policy with Deep Reinforcement Learning”** by Tian et al. The paper studies power allocation, which is also a sequential allocation problem with a simplex-constrained action space.

The paper argues that a Gaussian policy does not naturally satisfy simplex constraints. A common workaround is:

```text
Gaussian logits -> softmax -> simplex weights
```

But the paper points out two issues:

- The softmax mapping is not injective, because adding the same constant to all logits gives the same allocation.
- This can introduce bias and inefficient policy-gradient learning.

A Dirichlet policy instead samples directly on the simplex:

```text
a ~ Dirichlet(alpha)
```

So every sampled action is already a valid allocation.

**Connection To My Project**

Each CAOSD branch is itself a simplex. Therefore, a Dirichlet policy is a natural candidate for each branch allocation.

---

## 8. My Dirichlet Implementation

**Key Words**

- Autoregressive Dirichlet branch policy
- Branch weights, not logits
- Shared encoder
- Later branches condition on earlier branch weights
- Bounded concentration alpha

**Explanation**

I implemented a Dirichlet version of the autoregressive simplex policy.

The structure is:

```text
shared state encoder
-> Branch 1 Dirichlet
-> Branch 2 Dirichlet conditioned on Branch 1
-> Branch 3 Dirichlet conditioned on Branches 1-2
-> Branch 4 Dirichlet conditioned on Branches 1-3
-> CAOSD recombination
```


The deterministic evaluation action uses the Dirichlet mean:

```text
weight_i = alpha_i / sum(alpha)
```

This makes the policy simplex-native, but it introduces a new tuning problem: if alpha becomes too large, the policy becomes nearly deterministic and may stop exploring useful allocations.

---

## 9. Current Experiment Comparison: Four Simplex Policies

**Key Words**

- Four-policy comparison
- PPO versus RCPO
- Dirichlet versus Gaussian logits
- Fast early Dirichlet signal
- Stronger later Gaussian signal

**Evidence**

| Run | Policy | Best Validation Excess | Best Update | Latest Validation Excess | Main Observation |
|---|---:|---:|---:|---:|---|
| `simplex_v2.1_ppo_20260620_144053` | PPO Dirichlet | `+0.0343` | `709` | `+0.0031` | Finds a small early signal, then mostly loses it |
| `simplex_v2_ppo_20260612_195706` | PPO Gaussian | `+0.0739` | `15759` | `+0.0524` | Strongest PPO result, but needs long training |
| `simplex_v2.1_rcpo_20260620_144057` | RCPO Dirichlet | `+0.0539` | `1089` | `+0.0222` | Reaches useful validation excess quickly, then drops |
| `simplex_v2.1_rcpo_20260620_144107` | RCPO Gaussian | `+0.0630` | `6699` | `+0.0485` | Slower than Dirichlet, but more stable and higher later |

**Explanation**



```text
Dirichlet = faster early learning, weaker long-run stability
Gaussian  = slower learning, stronger later validation performance
```

This suggests that Dirichlet is still valuable, but likely needs better concentration scheduling, entropy control, or early stopping.

---

## 10. Dirichlet Behavior: Fast Early Learning, Then Decay

**Key Words**

- RCPO helps Dirichlet
- Best appears early
- Alpha already sharp
- Some components near upper bound
- Later training loses validation edge

**Evidence From RCPO Dirichlet**

At the best checkpoint:

```text
Run: simplex_v2.1_rcpo_none_20260620_144057
Best update: 1089
Best validation excess: +0.0539
Lambda at best: about 0.022
```

At update `1089`, the constraint was active but not dominant. The effective actor-level penalty was roughly 20-35% of the return-advantage scale across branches.

The learned Dirichlet concentrations were already quite sharp.

Branch 2:

```text
alpha mean: [29.0225, 0.8759, 7.8131]
branch weights: [0.7696, 0.0233, 0.2071]
```

Branch 3:

```text
alpha mean: [3.3215, 28.3249]
branch weights: [0.1049, 0.8951]
```

Branch 4:

```text
alpha mean: [1.8178, 11.7911, 12.3156, 0.9862, 28.2421, 1.5233]
branch weights: [0.0322, 0.2076, 0.2171, 0.0175, 0.4986, 0.0270]
```

**Explanation**

At the best point, Dirichlet has already learned strong allocation preferences. Some alpha values are close to the upper bound of `30`.

This explains why Dirichlet can find a good policy quickly. But it also explains why later training can become fragile: the policy may become too deterministic too early.

The current interpretation is that Dirichlet has a useful early discovery phase, but the concentration parameters need better control so the policy does not over-commit too quickly.

---

## 11. Gaussian Behavior: Slower But Stronger Later

**Key Words**

- Slower improvement
- PPO Gaussian strongest best validation
- RCPO Gaussian passes Dirichlet later
- Still KL-limited
- More robust late-training signal

**Evidence**

| Run | Policy | Best Validation Excess | Best Update | Latest Validation Excess | Observation |
|---|---:|---:|---:|---:|---|
| `simplex_v2_ppo_unconstrained_none_20260612_195706` | PPO Gaussian | `+0.0739` | `15759` | `+0.0524` | Highest peak, but long training |
| `simplex_v2.1_rcpo_none_20260620_144107` | RCPO Gaussian | `+0.0630` | `6699` | `+0.0485` | Constrained policy with stronger late stability |

Gaussian RCPO first passed the Dirichlet best threshold of `+0.0539` at:

```text
Update 2039
Validation excess: +0.05394
```

It reached maximum validation excess at:

```text
Update 6699
Validation excess: +0.06305
Lambda: about 0.225
```

**Explanation**

Gaussian RCPO learns more slowly than Dirichlet RCPO, but it keeps improving longer and eventually reaches a higher validation excess.

Gaussian PPO is also important because it shows that Gaussian logits can learn a strong simplex policy even without RCPO. However, the RCPO version is more aligned with the project goal because it includes the maximum-drawdown control layer.

This suggests:

- Dirichlet has faster early convergence.
- Gaussian has better later adaptability.
- Dirichlet may need better exploration or concentration scheduling.
- RCPO remains useful because the project is not only about return, but about return under drawdown control.

---

## 12. Main Interpretation From The Four Runs

**Key Words**

- Dirichlet is promising but fragile
- Gaussian is currently stronger
- RCPO improves the Dirichlet case
- PPO Gaussian is a strong baseline
- Need separate architecture effect from constraint effect

**Explanation**

The four-run comparison separates two questions:

```text
Question 1: Does Dirichlet help compared with Gaussian?
Question 2: Does RCPO help compared with PPO?
```

Current answer:

- Dirichlet helps with fast early discovery, especially under RCPO.
- Gaussian still performs better in later training.
- RCPO improves Dirichlet clearly compared with PPO Dirichlet.
- PPO Gaussian remains a strong baseline and should not be ignored.

The most important result is not simply “Dirichlet wins” or “Gaussian wins.” The more accurate conclusion is:

```text
Dirichlet learns a useful constrained allocation quickly,
but Gaussian currently gives stronger sustained validation performance.
```

This gives a clear next direction: keep Gaussian RCPO as the main working baseline, while tuning Dirichlet concentration and entropy to see whether it can keep its early advantage without collapsing later.

---

## 13. What I Learned

**Key Words**

- Branch credit matters
- Lambda scale matters
- Dirichlet is promising but fragile
- Gaussian is slower but more stable
- RCPO helps more than PPO

**Explanation**

The main lesson is that Simplex Decomposition changes the credit assignment problem. The policy is not one flat action anymore; it is four interacting branch policies.

The standalone branch-return design gives each branch a clearer training signal. This is a major improvement in the algorithm design.

The Dirichlet policy is theoretically attractive because it directly models simplex allocations. In my experiments, it does show fast early learning under RCPO. However, it tends to lose performance later, likely because the concentration parameters become too sharp.

Gaussian logits are less theoretically clean, but practically they still learn better in the current setup.

---

## 14. Current Limitations

**Key Words**

- Dirichlet drops after early success
- Concentration alpha can become too sharp
- Gaussian still KL-limited
- RCPO penalty still needs careful scaling
- Need stronger evaluation

**Explanation**

Current limitations:

- Dirichlet RCPO reaches a good policy quickly but does not preserve it.
- Dirichlet concentration parameters may saturate too early.
- Gaussian RCPO reaches a better maximum but needs many more updates.
- Gaussian RCPO is still KL-limited during training.
- The current evidence is still from synthetic markets, so the next step should include more robust evaluation and eventually real-market experiments.

---

## 15. Next Experiments

**Key Words**

- Dirichlet concentration schedule
- Lower max alpha
- Entropy tuning
- Compare best checkpoint, not latest only
- Real-market direction

**Possible Next Steps**

1. Tune Dirichlet concentration more carefully:

```yaml
dirichlet_init_concentration: 2.0
dirichlet_min_concentration: 0.3
dirichlet_max_concentration: 12.0
```

2. Add or increase entropy regularization for Dirichlet:

```yaml
entropy_coef: 0.0010
```

3. Use validation-based early stopping or stronger checkpoint selection because Dirichlet’s best policy appears early.

4. Continue Gaussian RCPO as the strongest practical baseline.

5. Move toward testing on real-market data after the synthetic experiment is understood.

---

## 16. Discussion Questions For Professor

**Key Words**

- Should Dirichlet remain main direction?
- Is early fast convergence enough?
- Should Gaussian be the practical baseline?
- How to present branch credit assignment?
- Real-market transition

**Questions**

1. Should I treat Dirichlet as the main policy architecture, or as an ablation inspired by simplex-native allocation?

2. Since Dirichlet reaches a strong policy earlier but later drops, should I focus on early stopping and concentration scheduling?

3. Is the standalone branch credit assignment a strong enough methodological contribution to emphasize?

4. Should the next stage prioritize real-market data, or should I first stabilize Dirichlet further on synthetic markets?

5. Should Gaussian logits remain the main working policy while Dirichlet is investigated as a theoretically motivated alternative?

---

## 17. One-Minute Summary

**Speaking Version**

Since the last meeting, I focused on two problems. First, I improved credit assignment for the four Simplex Decomposition branches. Instead of giving every branch the same final portfolio advantage, each branch now has its own standalone return and drawdown-cost signal, while lambda is still updated from the final combined portfolio. This makes the learning signal more consistent with the CAOSD structure.

Second, inspired by the Dirichlet power-allocation paper, I implemented an autoregressive Dirichlet branch policy. The connection is that each CAOSD branch is itself a simplex allocation problem. The Dirichlet policy is attractive because it samples directly on the simplex, unlike Gaussian logits followed by softmax.

Experimentally, RCPO Dirichlet reaches a relatively high validation excess quickly: about `+0.0539` at update `1089`. But it drops later. Gaussian RCPO improves more slowly, passes that level around update `2039`, and reaches a higher maximum of about `+0.0630` at update `6699`. My current interpretation is that Dirichlet gives faster early learning, but its concentration parameters become too sharp, while Gaussian is less theoretically clean but more robust in the current implementation.

---

## References

- Yuan Tian, Minghao Han, Chetan Kulkarni, Olga Fink, “A Prescriptive Dirichlet Power Allocation Policy with Deep Reinforcement Learning,” arXiv:2201.08445, 2022. https://arxiv.org/abs/2201.08445
- “Simplex Decomposition for Portfolio Allocation Constraints in Reinforcement Learning.”
