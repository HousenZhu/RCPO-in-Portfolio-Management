# V3.3: Long-Path Market Coverage Study

V3.3 tests the professor's hypothesis that fewer independent training markets
with longer histories can reduce the validation/test gap. It preserves the
V2.6 eight-asset market parameters, 252-step episodes, allocation constraints,
seed 0, 60,000 updates, rollout/minibatch sizes, and method-specific learning
rates.

## Shared evaluation design

Every V3.3 run is selected and evaluated on the same training-independent
continuation anchor and branch seeds. This is necessary because changing
`train_steps` would otherwise change the endpoint from which validation and
test markets continue.

- fixed evaluation anchor: seed `90000`, length `5040`;
- validation: 20 branches, seed base `100000`;
- test: 50 branches, seed base `200000`;
- validation selects checkpoints; test is used only for final reporting.

## Experiment matrix

### A. Market design ablation: Gaussian RCPO, epochs 3

| Config | Train markets | Steps per market | Purpose |
| --- | ---: | ---: | --- |
| `rcpo_gaussian_m8_s5040_e3.yaml` | 8 | 5,040 | Fresh V2.6 reference under shared evaluation |
| `rcpo_gaussian_m2_s5040_e3.yaml` | 2 | 5,040 | Isolate reducing market count |
| `rcpo_gaussian_m8_s10080_e3.yaml` | 8 | 10,080 | Isolate increasing path length |
| `rcpo_gaussian_m2_s10080_e3.yaml` | 2 | 10,080 | Professor's main setting |
| `rcpo_gaussian_m2_s20160_e3.yaml` | 2 | 20,160 | Test whether further length helps |

### B. Optimizer ablation

`rcpo_gaussian_m2_s10080_e4.yaml` differs from the main setting only by using
four optimization epochs. Comparing it with `...e3.yaml` isolates the Gaussian
epochs 3-versus-4 question.

### C. Method comparison on 2 markets x 10,080 steps

- `ppo_gaussian_m2_s10080_e4.yaml`: unconstrained simplex Gaussian PPO.
- `rcpo_dirichlet_m2_s10080_e4.yaml`: simplex Dirichlet RCPO.
- `rcpo_allocation_relative_drawdown_m2_s10080_e3.yaml`: non-simplex soft
  allocation plus relative-drawdown RCPO baseline.
- `cf_reward_gaussian_m2_s10080_e3.yaml`: counterfactual-reward Gaussian RCPO.
- `cf_reward_dirichlet_m2_s10080_e4.yaml`: counterfactual-reward Dirichlet RCPO.

The other V3.2 counterfactual cost combinations are intentionally omitted.
They do not directly answer the market-coverage question and would weaken the
controlled comparison.

## Commands

```bash
# Market ablation
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_gaussian_m8_s5040_e3.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_gaussian_m2_s5040_e3.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_gaussian_m8_s10080_e3.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_gaussian_m2_s10080_e3.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_gaussian_m2_s20160_e3.yaml

# Epoch ablation
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_gaussian_m2_s10080_e4.yaml

# Method comparison
python train.py --algo ppo_unconstrained --config configs/v3.3_experiment2_8assets/ppo_gaussian_m2_s10080_e4.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/rcpo_dirichlet_m2_s10080_e4.yaml
python train.py --algo rcpo --constraint-allocation-drawdown --config configs/v3.3_experiment2_8assets/rcpo_allocation_relative_drawdown_m2_s10080_e3.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/cf_reward_gaussian_m2_s10080_e3.yaml
python train.py --algo rcpo --constraint-drawdown --config configs/v3.3_experiment2_8assets/cf_reward_dirichlet_m2_s10080_e4.yaml
```

Run the market-ablation group first. If compute is limited, prioritize the
four `2x5040`, `8x10080`, `2x10080-e3`, and `2x10080-e4` runs because the
existing V2.6 result already provides supporting evidence for `8x5040-e3`.
