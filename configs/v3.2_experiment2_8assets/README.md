# V3.2 Counterfactual Branch Credit

V3.2 keeps the V2.6 Experiment 2 market, seed 0, CAOSD constraints,
relative-current-drawdown RCPO, dynamic alpha, and mean-step lambda update.
Only branch credit changes.

The six runs form a 3 x 2 comparison:

- counterfactual reward with global actual cost;
- standalone reward with counterfactual cost;
- counterfactual reward with counterfactual cost;
- each tested with autoregressive Gaussian and Dirichlet policies.

Counterfactuals are stateful open-loop paths. One branch is replaced by its
uniform neutral action, realized downstream branch actions are held fixed, and
the complete CAOSD mapping is rerun. Lambda is always updated from the actual
final portfolio cost, never from signed counterfactual cost differences.

The 11,400-update budget is the approximately 13-hour pilot budget, scaled from
the established 14,000-update / 16-hour estimate. Learning-rate endpoints match
the first 11,400 updates of the original V2.6
60,000-update schedules.

```powershell
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/v3.2_experiment2_8assets/counterfactual_reward_gaussian.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/v3.2_experiment2_8assets/counterfactual_cost_gaussian.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/v3.2_experiment2_8assets/counterfactual_reward_cost_gaussian.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/v3.2_experiment2_8assets/counterfactual_reward_dirichlet.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/v3.2_experiment2_8assets/counterfactual_cost_dirichlet.yaml
py -3.11 train.py --algo rcpo --constraint-drawdown --config configs/v3.2_experiment2_8assets/counterfactual_reward_cost_dirichlet.yaml
```
