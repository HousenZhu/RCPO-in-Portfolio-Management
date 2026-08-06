# Experiment 2: Eight-Asset Difficult Environment

This folder contains the 8-risky-asset plus cash experiment. Experiment 1
configuration files in `configs/` remain unchanged.

## Why This Environment Is Harder

The full portfolio contains cash at index `0` and risky assets at indices
`1..8`. The two allocation constraints are:

```text
V1 = [1, 4, 6], minimum weight 0.50
V2 = [2, 4, 7], minimum weight 0.40
V1 intersection V2 = [4]
```

Asset 4 is deliberately weak in both regimes. It can satisfy both allocation
constraints, but concentrating there sacrifices expected return. The intended
regime specialists are:

- Low volatility: assets 1 and 2, one in each constraint group.
- High volatility: assets 6 and 7, one in each constraint group.
- Asset 8: strongest high-volatility opportunity, outside both groups.

This forces the policy to learn a regime-dependent feasible allocation instead
of placing nearly all weight in one high-return overlap asset.

The active thresholds are deliberately asymmetric and sum to `0.90`. Therefore
CAOSD has `z1 = max(0, 0.50 + 0.40 - 1) = 0`, so the singleton intersection
branch does not receive mandatory mass. A policy may still use asset 4, but it
must choose it because the allocation is useful rather than because both
constraints mechanically force it.

The 20-step observation window is retained. With momentum decay `0.94`, its
effective memory is roughly 17 trading steps, so increasing the lookback would
add input size without a clear signal benefit. The observation dimension grows
from 122 to 191, so the shared policy encoder is changed only slightly from
`[128, 128]` to `[192, 128]`.

For simplex policies, the CAOSD branch sizes are `[1, 3, 3, 9]`, giving action
dimension 16. Softmax policies have action dimension 9.

For the softmax RCPO baselines, uniform initial weights produce an allocation
constraint cost of roughly `0.00694` with `allocation_constraint_cost_scale:
20.0`. With three optimization epochs and `lambda_lr_up: 0.001`, the initial
lambda increase is roughly `2.05e-5` per update, keeping the constraint visible
without immediately overwhelming the return advantage.

## Training Commands

```powershell
py -3.11 train.py --algo ppo_unconstrained --config configs/experiment2_8assets/simplex_ppo_gaussian.yaml

py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/experiment2_8assets/simplex_rcpo_gaussian.yaml

py -3.11 train.py --algo ppo_unconstrained --config configs/experiment2_8assets/simplex_ppo_dirichlet.yaml

py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/experiment2_8assets/simplex_rcpo_dirichlet.yaml

py -3.11 train.py --algo rcpo --constraint-allocation --config configs/experiment2_8assets/rcpo_allocation_penalty.yaml

py -3.11 train.py --algo rcpo --constraint-allocation-drawdown --config configs/experiment2_8assets/rcpo_allocation_drawdown_penalty.yaml
```

All six configurations inherit shared environment settings from `base.yaml`.
Each run still saves a complete resolved `config_snapshot.yaml`.
