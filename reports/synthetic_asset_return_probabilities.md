# Synthetic Asset Return Probabilities

This note summarizes the one-day return distribution for each risky asset under the current synthetic market config in `configs/default.yaml`.

The calculation ignores the momentum adjustment so the base regime signal is easier to see. In the actual simulator, the mean is shifted by `momentum_strength * clip(trend, -momentum_clip, momentum_clip)`.

Formula used:

```text
asset_return_t ~ Normal(mean_regime_asset, std_regime_asset)
mean_regime_asset = regime_base_drift + asset_offset + regime_drift_adjustment
std_regime_asset = regime_vol_scale * asset_vol_multiplier
```

Current shared settings:

- Number of risky assets: `5`
- Asset drift offsets: `[-0.0002, -0.0001, 0.0, 0.0001, 0.0002]`
- Asset volatility multipliers: `[0.9, 0.95, 1.0, 1.05, 1.1]`
- Momentum strength: `0.2`
- Momentum decay: `0.94`
- Momentum clip: `0.01`

## Low-vol regime

- Base drift: `0.000700`
- Volatility scale: `0.0080`
- Pairwise correlation: `0.15`

| Asset | Mean Daily Return | Daily Std | P(Return > 0) | P(Return < 0) |
|---:|---:|---:|---:|---:|
| 1 | `0.130%` | `0.720%` | `57.16%` | `42.84%` |
| 2 | `0.125%` | `0.760%` | `56.53%` | `43.47%` |
| 3 | `0.030%` | `0.800%` | `51.50%` | `48.50%` |
| 4 | `0.035%` | `0.840%` | `51.66%` | `48.34%` |
| 5 | `0.045%` | `0.880%` | `52.04%` | `47.96%` |

## High-vol regime

- Base drift: `-0.000400`
- Volatility scale: `0.0200`
- Pairwise correlation: `0.55`

| Asset | Mean Daily Return | Daily Std | P(Return > 0) | P(Return < 0) |
|---:|---:|---:|---:|---:|
| 1 | `-0.160%` | `1.800%` | `46.46%` | `53.54%` |
| 2 | `-0.130%` | `1.900%` | `47.27%` | `52.73%` |
| 3 | `-0.100%` | `2.000%` | `48.01%` | `51.99%` |
| 4 | `0.010%` | `2.100%` | `50.19%` | `49.81%` |
| 5 | `0.040%` | `2.200%` | `50.73%` | `49.27%` |

## Interpretation

- Low-vol regime makes assets 1 and 2 the clearest winners by expected daily return.
- High-vol regime makes assets 4 and 5 the defensive winners, while assets 1 to 3 have negative expected daily return.
- Even winner assets can have many negative days because daily volatility is much larger than daily drift.
- The RL policy should not expect every winning asset to rise each day; it should learn regime-dependent tilts from repeated evidence over the observation window.
