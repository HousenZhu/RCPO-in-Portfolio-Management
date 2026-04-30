# RCPO Portfolio Project: Current Meeting Handout

## 1. What Changed Since Last Meeting

**Key Idea**

The project has moved from downside-risk RCPO toward a more realistic portfolio safety constraint: maximum drawdown relative to an online equal-weight benchmark. I also added noisy-reward experiments to test whether PPO, RCPO, and GDRC can still learn when the reward signal is corrupted during training.

**Main Updates**

- RCPO now uses a benchmark-relative maximum drawdown constraint instead of downside or Sortino constraints.
- Training can optionally corrupt the reward channel with Gaussian noise.
- Evaluation remains clean, so validation/test returns still reflect real portfolio performance.
- GDRC is now tested in the noisy-reward setting, where reward correction has a clearer purpose.
- The current comparison uses six runs: clean PPO/RCPO, noisy PPO/RCPO, and noisy PPO/RCPO with GDRC.

[Figure Placeholder: Environment and portfolio step loop]

Suggested sources:

- `runs/new_ppo_unconstrained_none_20260424_170155/seed_0/evaluation/training_return.png`
- `runs/new_rcpo_none_20260427_184555/seed_0/evaluation/training_return.png`

---

## 2. Current Environment And Market Setup

**Key Idea**

The environment is a long-only portfolio simulator with cash plus five risky assets. The policy chooses portfolio weights, the environment applies transaction costs, and the reward is net portfolio log return.

**Technical Details**

- Assets: cash plus five risky assets.
- Action: policy outputs allocation logits.
- Weights: logits are converted to long-only weights by softmax.

```text
weights_t = softmax(action_logits_t)
```

- Risky-asset return:

```text
raw_return_t = dot(weights_risky_t, risky_asset_returns_t)
```

- Turnover and transaction cost:

```text
turnover_t = sum(abs(weights_t - weights_{t-1}))
transaction_cost_t = transaction_cost_rate * turnover_t
```

- Net return and clean reward:

```text
net_return_t = raw_return_t - transaction_cost_t
true_reward_t = log(1 + net_return_t)
```

**Synthetic Market**

- The market has persistent low-volatility and high-volatility regimes.
- Low-volatility regimes favor assets 1 and 2.
- High-volatility regimes favor more defensive assets 4 and 5.
- Mild momentum makes recent trends informative.
- Correlation is regime-dependent: lower in low-volatility regimes and higher in high-volatility regimes.
- The regime label is hidden, so the policy must infer market state from returns, rolling means, and volatility.

---

## 3. New RCPO Constraint: Maximum Drawdown

**Key Idea**

The new RCPO constraint is not simply “avoid negative returns.” It asks whether the learned portfolio’s worst peak-to-trough loss is worse than an online equal-weight benchmark budget.

**Why This Constraint Is More Meaningful**

- A good real-market policy should not only chase return; it should control large portfolio losses.
- Maximum drawdown is easy to explain: it is the largest fall from a previous peak.
- Comparing drawdown against equal weight makes the constraint relative to a simple baseline instead of an arbitrary fixed number.

[Figure Placeholder: Online drawdown constraint calculation]

Caption suggestion: show the model portfolio value, running peak, current drawdown, equal-weight benchmark drawdown, and the effective drawdown budget.

---

## 4. How The Constraint Cost Is Calculated

**Step 1: Track model portfolio value**

```text
portfolio_value_t = portfolio_value_{t-1} * (1 + net_return_t)
```

**Step 2: Track the running peak**

```text
running_peak_t = max(running_peak_{t-1}, portfolio_value_t)
```

**Step 3: Compute current drawdown**

```text
current_drawdown_t = (running_peak_t - portfolio_value_t) / running_peak_t
```

**Step 4: Track maximum drawdown so far**

```text
max_drawdown_t = max(previous_max_drawdown, current_drawdown_t)
```

**Step 5: Track equal-weight benchmark drawdown online**

The environment also tracks an equal-weight portfolio inside the same episode. This benchmark uses the same market path and transaction-cost model, but it does not require running a second environment.

```text
equal_weight_max_drawdown_t = max drawdown of equal weight up to time t
```

**Step 6: Define the effective drawdown budget**

```text
budget_t = max(
    drawdown_budget_floor,
    benchmark_drawdown_margin * equal_weight_max_drawdown_t
)
```

In the current runs:

```yaml
drawdown_budget_floor: 0.05
benchmark_drawdown_margin: 0.96
drawdown_cost_scale: 0.01
```

This means the model is allowed at least a 5% drawdown budget, but once equal weight has a larger drawdown, the budget becomes 96% of equal weight's max drawdown.

**Step 7: Compute violation and cost**

```text
drawdown_violation_t = max(0, max_drawdown_t - budget_t)
constraint_cost_t = drawdown_violation_t^2 / drawdown_cost_scale
```

**Step 8: RCPO uses this cost through the Lagrange multiplier**

```text
A_RCPO = A_reward - lambda * A_cost
```

If the batch constraint cost is above the target `alpha`, lambda increases. If the batch cost is below alpha, lambda decreases.

Current alpha is dynamic:

```text
alpha_t = ((alpha_budget_ratio * effective_drawdown_budget_t)^2) / drawdown_cost_scale
```

With:

```yaml
alpha_budget_ratio: 0.05
```

This means RCPO tolerates a small average violation equal to about 5% of the current drawdown budget.

---

## 5. How Gaussian Reward Noise Is Added

**Key Idea**

Reward noise is added only during training. The environment still computes the true clean portfolio reward, but the policy update can receive a noisy observed reward.

**Formula**

```text
true_reward_t = log(1 + clean_net_return_t)
observed_reward_t = true_reward_t + Normal(0, reward_noise.std)
```

Current noisy-reward setting:

```yaml
reward_noise:
  enabled: true
  mode: gaussian
  std: 0.003
```

**Important Separation**

- PPO and RCPO train on the observed reward when noise is enabled.
- Drawdown constraint cost is still clean.
- Validation and test evaluation use clean portfolio returns.
- Checkpoint selection still uses clean validation performance versus equal weight.
- Equal-weight comparisons are clean.

This design isolates reward-channel corruption: only the reward signal given to learning is noisy.

---

## 6. How GDRC Tries To Correct Noisy Rewards

**Key Idea**

GDRC is a reward-correction module. It tries to learn a mapping from transition data to reward so that training can use a corrected reward instead of the raw noisy observed reward.

**Input And Target**

```text
(observation_t, action_t, next_observation_t) -> observed_reward_t
```

The reward is discretized into bins, and the reward critic predicts which bin the transition belongs to.

**Current Stabilized GDRC Settings**

```yaml
reward_correction:
  hidden_sizes: [256, 256]
  learning_rate: 0.0005
  train_epochs_per_update: 5
  gdrc_candidate_bins: [48, 64]
  gdrc_vote_decay: 0.95
  gdrc_range_window_updates: 20
  gdrc_range_percentiles: [1.0, 99.0]
  correction_delta_clip: 0.0015
```

The important change is that GDRC no longer uses very coarse bins like 4 bins. In earlier noisy runs, coarse bins caused reward corrections that were too large and biased.

**Correction Rule**

```text
raw_delta = GDRC suggested reward correction
effective_delta = clip(raw_delta, -0.0015, 0.0015)
corrected_reward_t = observed_reward_t + effective_delta
```

This keeps the correction bounded. With reward noise standard deviation `0.003`, a clip of `0.0015` means GDRC can adjust at most half of one noise standard deviation per step.

[Figure Placeholder: GDRC reward correction diagnostics]

Suggested sources:

- `runs/noise_v1_rcpo_gdrc_20260429_121506/seed_0/evaluation/training_reward_correction.png`
- `runs/noise_v1_rcpo_gdrc_20260429_121506/seed_0/evaluation/gdrc_selected_bins.png`
- `runs/noise_v1_ppo_unconstrained_gdrc_20260429_153756/seed_0/evaluation/training_reward_correction.png`

---

## 7. Six-Experiment Comparison

**Key Idea**

The six experiments compare clean training, noisy reward training, and noisy reward correction. The main question is whether RCPO and GDRC help preserve out-of-sample performance when training rewards are noisy.

| Run | Setting | Status | Best Validation Excess | Best Test Excess | Validation Win Rate | Test Win Rate | Main Observation |
|---|---:|---:|---:|---:|---:|---:|---|
| `new_ppo_unconstrained_none_20260424_170155` | Clean PPO | Complete | `+0.1046` | `+0.0610` | `0.8` | `0.6` | Strong clean baseline; high validation and positive test excess. |
| `new_rcpo_none_20260427_184555` | Clean RCPO | Complete | `+0.1069` | `+0.0194` | `1.0` | `0.6` | Strong validation with better drawdown control, but lower test excess than clean PPO. |
| `noise_ppo_unconstrained_none_20260428_183102` | Noisy PPO | Complete | `+0.0481` | `-0.0035` | `1.0` | `0.4` | Noisy reward hurts generalization; validation improves but test is slightly below equal weight. |
| `noise_rcpo_none_20260428_183113` | Noisy RCPO | Complete | `+0.0937` | `+0.0270` | `1.0` | `0.6` | RCPO is more robust than PPO under noisy rewards, but later checkpoints degrade. |
| `noise_v1_ppo_unconstrained_gdrc_20260429_153756` | Noisy PPO + GDRC | Complete | `+0.0212` | `-0.0145` | `0.8` | `0.4` | Completed to `12000` updates; GDRC improved validation versus the earlier partial-training snapshot, but test excess stayed below equal weight. |
| `noise_v1_rcpo_gdrc_20260429_121506` | Noisy RCPO + GDRC | Complete | `+0.1014` | `+0.0411` | `1.0` | `0.4` | Best noisy-reward test excess so far; GDRC appears more useful with RCPO than with PPO. |

Notes:

- "Excess" means model final cumulative return minus equal-weight final cumulative return across future branches.
- Best validation checkpoint is used for completed-run test reporting.
- The PPO+GDRC result is now complete, but it is not a positive generalization result because the best-validation checkpoint still has negative test excess.

[Figure Placeholder: Clean PPO versus clean RCPO validation/test comparison]

Suggested sources:

- `runs/new_ppo_unconstrained_none_20260424_170155/seed_0/evaluation_best/mean_cumulative_return_test.png`
- `runs/new_rcpo_none_20260427_184555/seed_0/evaluation_best/mean_cumulative_return_test.png`

[Figure Placeholder: Noisy reward experiment comparison]

Suggested sources:

- `runs/noise_ppo_unconstrained_none_20260428_183102/seed_0/evaluation_best/mean_cumulative_return_test.png`
- `runs/noise_rcpo_none_20260428_183113/seed_0/evaluation_best/mean_cumulative_return_test.png`
- `runs/noise_v1_ppo_unconstrained_gdrc_20260429_153756/seed_0/evaluation_best/mean_cumulative_return_test.png`
- `runs/noise_v1_rcpo_gdrc_20260429_121506/seed_0/evaluation_best/mean_cumulative_return_test.png`

---

## 8. Main Observations

**Observation 1: Clean PPO and clean RCPO both learn strong policies.**

Clean PPO and clean RCPO both achieved about `+0.10` validation excess. PPO had stronger best test excess in this set, while RCPO had a drawdown-aware objective and a strong validation result.

**Observation 2: Noisy rewards hurt PPO more clearly.**

Noisy PPO without correction reached positive validation excess but slightly negative test excess. This suggests PPO can learn noisy validation behavior but generalization becomes weaker.

**Observation 3: RCPO is more robust under noisy rewards.**

Noisy RCPO without GDRC achieved stronger validation and positive test excess compared with noisy PPO. However, the later checkpoint degraded, which means checkpoint selection remains important.

**Observation 4: GDRC needs stabilization and does not automatically help PPO.**

Earlier GDRC versions over-corrected rewards because they selected very coarse bins. The current stabilized version uses 48/64 bins and clips reward corrections.

In the completed PPO+GDRC run, best validation excess improved to `+0.0212`, but best test excess was `-0.0145`. This means the correction module can change the learned policy, but for PPO alone it did not yet improve out-of-sample performance.

**Observation 5: RCPO+GDRC is the strongest noisy-reward result so far.**

The current noisy RCPO+GDRC run achieved best validation excess `+0.1014` and test excess `+0.0411`. That is better than noisy PPO and noisy RCPO without correction on test excess.

**Observation 6: The useful signal is the interaction between RCPO and GDRC.**

The current evidence suggests GDRC is more useful when paired with the drawdown-aware RCPO objective than when added to PPO alone. This should still be treated as a single-seed result, but it is the clearest direction for the next experiment.

---

## 9. Limitations

**Current Limitations**

- Results are still mostly single-seed comparisons.
- Validation and test use synthetic future branches, not real market data.
- RCPO+GDRC improves noisy-reward test excess, but test win rate is still only `0.4` in the current best checkpoint.
- Turnover remains high for several strong-return policies.
- GDRC correction quality is hard to judge from reward loss alone; validation performance and correction bias are more important.

**Important Diagnostic For GDRC**

`reward_oce` is the ordinal cross-entropy loss of the reward critic. Lower is better, but it is not the portfolio reward.

The better RCPO+GDRC run showed:

```text
gdrc_selected_bins = 64
reward_correction_delta_abs_mean ≈ 0.00117
reward_correction_delta_mean near 0
```

This means GDRC is applying bounded corrections without a large systematic positive or negative bias.

---

## 10. Next Experiments

**Next Step 1: Analyze why PPO+GDRC did not generalize.**

The PPO+GDRC noisy run is now complete. It improved validation modestly, but the best-validation checkpoint still had negative test excess, so the next step is to inspect whether the reward correction is adding useful signal or just extra policy variance.

**Next Step 2: Repeat across multiple seeds.**

The current comparison is useful, but final conclusions should use several random seeds.

**Next Step 3: Compare best and last checkpoints.**

Some RCPO runs have strong best checkpoints but weaker final checkpoints. This should be explicitly analyzed.

**Next Step 4: Tune GDRC correction strength carefully.**

The current safer setting is:

```yaml
correction_delta_clip: 0.0015
```

Larger clips, such as `0.0045`, allowed larger corrections but performed worse in the current comparison.

**Next Step 5: Add turnover-aware evaluation or constraint.**

High turnover is still a practical concern. It may need to become either a soft cost, a constraint, or a reporting threshold.

---

## 11. One-Minute Summary

This project formulates portfolio management as a constrained reinforcement learning problem. PPO learns to maximize net portfolio log return, while RCPO also controls maximum drawdown relative to an online equal-weight benchmark. I then added Gaussian noise to the training reward to test whether the algorithms can still learn from corrupted reward feedback. Clean PPO and RCPO both learn strong policies. Under noisy rewards, PPO generalization weakens, while RCPO is more robust. GDRC was added as a reward-correction layer; after stabilizing it with fine bins and clipped corrections, RCPO+GDRC currently gives the strongest noisy-reward test result among the noisy experiments, while PPO+GDRC still does not beat equal weight on test.
