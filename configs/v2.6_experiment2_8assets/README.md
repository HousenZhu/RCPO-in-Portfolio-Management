# V2.6 Experiment 2: Eight-Asset Configuration Set

This folder is the authoritative six-run configuration set for the V2.6 Phase 1 experiment. All runs use the same eight-risky-asset synthetic market, 252-step episodes, constrained-neutral initialization and benchmark, ten validation branches, twenty final test branches, and 60,000 updates.

## Shared V2.6 Changes

- Active drawdown risk uses benchmark-relative current drawdown, not repeated maximum-drawdown occupancy.
- The online budget is `max(0.05, 0.90 * benchmark_current_drawdown)`.
- Dynamic alpha uses the same cost units as the active drawdown cost.
- The observation includes current/max drawdown for the agent and benchmark, current budget, drawdown gap, and episode progress.
- Simplex branch return credit remains standalone, while RCPO branches share the actual final-portfolio cost advantage.
- Lambda updates once per rollout.
- PPO uses non-negative KL estimation and rejects an over-limit minibatch before backpropagation.
- Metrics and validation records use schema v2 with separate `metrics.jsonl` and `validation_metrics.jsonl`.

## Parameter Rationale From V2.3-V2.5

V2.4 and V2.5 showed that lowering Gaussian learning rates sharply reduced the excessive KL early-stop rate. V2.5 also showed that RCPO Gaussian could become cost-dominated, so V2.6 lowers its actor learning rate to `6e-5`, uses three epochs, and slows upward lambda movement to `7.5e-4` once per rollout. The faster `0.01` downward rate lets constraint pressure relax after the rollout returns below alpha.

Dirichlet runs rarely hit the KL limit but changed too slowly and could approach deterministic behavior. V2.6 therefore uses a broader but bounded concentration range `[0.5, 8.0]`, starts at `1.5`, and keeps slightly higher entropy. PPO Dirichlet receives `1e-4`; RCPO Dirichlet uses `7.5e-5`.

| Run | Initial LR | Final LR | Epochs | Entropy |
|---|---:|---:|---:|---:|
| Simplex PPO Gaussian | 0.000085 | 0.000040 | 4 | 0.0015 |
| Simplex RCPO Gaussian | 0.000060 | 0.000030 | 3 | 0.0015 |
| Simplex PPO Dirichlet | 0.000100 | 0.000040 | 4 | 0.0020 |
| Simplex RCPO Dirichlet | 0.000075 | 0.000030 | 4 | 0.0020 |
| Soft allocation RCPO | 0.000085 | 0.000030 | 3 | 0.0010 |
| Allocation + relative drawdown RCPO | 0.000060 | 0.000025 | 3 | 0.0010 |

The four simplex policies enforce allocation constraints exactly through CAOSD. The two softmax RCPO baselines test the same allocation requirements without hard simplex feasibility.

Older files in `configs/experiment2_8assets/` remain frozen V2.5 references. Root-level one-file configs remain legacy convenience files and are not used by V2.6.

