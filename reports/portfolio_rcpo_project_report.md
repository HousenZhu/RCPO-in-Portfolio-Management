# RCPO Portfolio Management Project: Shared Meeting Handout

## Synthetic Market


The market generator used a two-regime Gaussian process. One regime has lower volatility and more positive drift. The other has higher volatility and more negative drift. 
```text
[[0.95, 0.05],
 [0.10, 0.90]]
```

The regime label is hidden from the policy.


- Low-volatility regime: lower volatility and generally stronger drift.
- High-volatility regime: higher volatility and weaker or negative drift.
- Mild momentum gives recent trends some predictive value.
- Correlation is lower in low-volatility regimes and higher in high-volatility regimes.


---

## Environment

long-only portfolio simulator with cash, five risky assets, transaction costs

The agent observes recent market and portfolio features, outputs allocation logits, and receives net log return after transaction cost.



The **observation** includes risky-asset returns over the lookback window, rolling mean, rolling volatility, current weights, and previous turnover. 

The policy action is converted into weights:

```text
action_weights = softmax(action_logits)
```

The raw simple portfolio return is the weighted risky-asset return:

```text
raw_return_t = dot(w_risky, asset_returns_t)
```

Turnover is the total absolute weight change:

```text
turnover_t = sum(abs(w_t - w_{t-1}))
```

Transaction cost is proportional to turnover:

```text
transaction_cost_t = transaction_cost_rate * turnover_t
```

The net simple return and reward are:

```text
net_return_t = raw_return_t - transaction_cost_t
reward_t = log(1 + net_return_t)
```


- Cash return is treated as zero in this prototype.
- Reward is net log return, which is additive over time.


---

## PPO 

PPO is the return-only that maximizes portfolio reward without constraints.


`runs/latest_ppo_unconstrained_20260416_220423/seed_0/evaluation/training_return.png`

---

## RCPO Algorithm


```text
A_reward = GAE(reward)
A_cost   = GAE(cost)
```

The policy advantage becomes:

```text
A_RCPO = A_reward - lambda * A_cost
```

The Lagrange multiplier is dynamic during training

```text
lambda = max(0, lambda + lambda_lr * (J_C_batch - alpha))
```

If cost is below the budget, lambda decreases to zero. 


## Downside Constraints

implements two RCPO constraint modes: downside-risk cost and Sortino-ratio target violation.

**Explanation**

downside risk - penalizes negative net returns:

```text
downside_cost_t = max(0, -net_return_t)^2
normalized_downside_cost_t = downside_cost_t / downside_scale
```

- The current downside RCPO version can optionally include diversification cost through concentration, but this is being tuned carefully.

---

## DRC And GDRC

reward-correction modules for the reward stream used for learning

distributional reward critic to predict observed reward from transition:

```text
(obs_t, action_t, next_obs_t) -> reward_t
```

Rewards are discretized into bins. The critic predicts a reward bin, and the corrected reward is computed by comparing the predicted bin with the observed bin:

```text
corrected_reward = observed_reward
                 + (predicted_label - observed_label) * bin_width
```

GDRC extends reward critics with different bin counts


---

## Development Path

The first version was long-only RCPO prototype with a synthetic two-regime market and downside-risk constraint.

The second phase introduced a richer hybrid long/short environment 


The third phase returned to long-only allocation and added group-allocation diagnostics. Group A was risky assets 1 and 2, and Group B was risky assets 3 to 5. 
---
- `latest_rcpo_20260330_141201`
latest_rcpo_20260409_182536

The fourth phase restored a clean train, validation, and test workflow. Validation was used for checkpoint selection, while test was reserved for final reporting.
---
latest_rcpo_20260417_183003
latest_ppo_unconstrained_20260416_220423

The fifth phase added Sortino RCPO constraint mode.
---
latest_rcpo_20260417_211231

- Sortino constraints can be noisy because Sortino is estimated from rolling returns.

The sixth phase added DRC/GDRC reward correction.
---
latest_rcpo_gdrc_20260419_173123

- PPO was stable but conservative.
- RCPO was more aggressive and often higher turnover.
- GDRC did not really solve something because of clean reward.
- Single validation/test branches were not enough for robust conclusions.

The first problem was overfitting. The model could look good on the market path it saw during training or validation, but not consistently beat equal weight on future branches.

The second problem was high turnover. the policy frequently changed a very large fraction of the portfolio. High turnover can make the strategy unrealistic and unstable

The third problem was validation fragility. A single validation market can make one checkpoint look strong by chance. This motivated multi-branch validation.

The seventh phase added the robust multi-market upgrade: multiple training markets, multi-branch validation, equal-weight initialization, learnable market structure, and optional diversification pressure.
---

The current improvement direction is to train on many markets and select checkpoints by average future performance versus equal weight.

---



The next experiments should be ablations, not just longer training runs. The clean comparison is PPO versus downside-only RCPO versus RCPO with optional reward correction, all under the same multi-branch validation setup.



