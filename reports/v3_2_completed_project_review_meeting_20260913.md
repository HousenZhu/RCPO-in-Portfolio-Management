# V3.2 Completed Experiments and Project Review

**Shared meeting handout | Housen Zhu | 13 September 2026**

[Open the 13-slide Google Slides presentation](https://docs.google.com/presentation/d/1SJvuVsIeWNjRo5Ru1SpEmhXTSdAPzbYV8yaa0R9sGhk/edit)

Suggested duration: 20-25 minutes. The slide text is concise; the speaking scripts below provide a fuller explanation. Results reflect existing completed artifacts; this task did not rerun or modify training.

## Executive Summary

All six V3.2 runs completed 60,000 updates. Counterfactual-cost Gaussian offers the clearest risk-side improvement over its matched V2.6 control. Counterfactual-reward Dirichlet has the highest observed relative wealth, but only a small gain over its control and substantially greater turnover. Hard allocation feasibility is maintained, while future drawdown feasibility remains unresolved. A complete LaTeX manuscript draft is available for professor review.

## 1. Constraint-Aware Portfolio RL

**Key Points**
- Completed V3.2 experiments
- Project evaluation
- Final manuscript and discussion

**Speaking Script**

Since our last meeting, I have completed the continuation of all six V3.2 experiments to sixty thousand updates, evaluated them against the matched V2.6 controls, and written the final report draft. Today I would like to focus on what the completed evidence actually supports. The strongest result remains the separation between hard allocation feasibility and learned risk control. Counterfactual credit changes the trade-off, but it does not solve every remaining weakness. I will first recap the experiment, then discuss the results and limitations, and finally ask for feedback on the report framing.

## 2. Progress since the last meeting

**Key Points**
- Six V3.2 runs: 11,400 to 60,000 updates each
- Final evaluation: 20 shared unseen continuation markets
- Project assessment and final_project_report.tex draft

**Speaking Script**

The first change is that the earlier V3.2 results are no longer interim results. All six runs have now completed sixty thousand updates. This gives the counterfactual variants the same environment-interaction budget as the V2.6 controls. The second change is a shared final evaluation, rather than comparing isolated training curves. The third is the manuscript: I have organised the project into problem formulation, method, experiments, and limitations. These three pieces let us distinguish a completed implementation from a demonstrated improvement. The training summaries verify completion; the comparison summaries are the source of the results shown today.

## 3. Three distinct parts of the problem

**Key Points**
- Hard: two overlapping minimum-allocation constraints
- Soft: benchmark-relative current-drawdown cost
- Credit: which branch improved the final portfolio?

**Speaking Script**

The project has three layers. First, the CAOSD mapping guarantees the two allocation requirements for each action. In this experiment they are fifty percent in assets one, four, and six, and forty percent in assets two, four, and seven. Second, RCPO addresses path-dependent drawdown. Its budget is the larger of five percent and ninety percent of the constrained-neutral benchmark current drawdown. The cost is the squared positive excess divided by zero point one. Third, branch credit is a learning question: after the branch actions are combined, how should each branch receive credit? V3.2 changes that third layer while keeping the hard map and global risk target.

## 4. V3.2 tests three credit rules, twice

**Key Points**
- CF reward: counterfactual reward + global actual cost
- CF cost: standalone reward + counterfactual cost
- CF reward + cost: both branch signals are counterfactual

**Speaking Script**

For each rule I train both an autoregressive Gaussian-logit actor and an autoregressive Dirichlet actor. The counterfactual replaces one branch with its neutral allocation, keeps the other realised branch actions fixed, and recomputes the full CAOSD map. Each alternative has its own turnover and wealth history on the same market returns. The reward or cost difference is actual minus counterfactual. Importantly, the reward-only variant retains global actual cost, while the cost-only variant retains standalone reward. The combined variant changes both signals. A single global lambda still updates from actual portfolio cost, not from signed counterfactual differences. Because downstream actions are held fixed, this is an open-loop credit heuristic, not an unbiased causal effect.

## 5. A matched comparison, with limits

**Key Points**
- 8 training markets; 8 risky assets + cash; 252-step episodes
- 60,000 x 2,048 transitions per policy; one training seed
- 10 validation branches for selection; 20 shared test branches

**Speaking Script**

The market contains eight risky assets and cash. Training uses eight synthetic paths, and each episode lasts two hundred and fifty-two trading steps. Every policy has sixty thousand rollouts of two thousand and forty-eight transitions, which is about one hundred and twenty-three million environment transitions. The validation continuations select the checkpoint; the twenty separate test continuations evaluate it. Gaussian and Dirichlet use different numbers of optimisation epochs, so equal interaction budget should not be described as equal computation. Most importantly, the twenty markets are not twenty independent training seeds. They describe future-path variation conditional on a single trained seed. Relative wealth is calculated within each market as model wealth divided by baseline wealth minus one, and only then averaged.

## 6. Completed future-market return comparison

**Key Points**
- Positive relative wealth for all six selected V3.2 policies
- Dirichlet leads the return ranking
- Return ranking alone is not a risk-control result

**Speaking Script**

This figure shows the mean relative-wealth path on the same twenty future markets. The horizontal zero line is the constrained-neutral baseline. The selected V3.2 policies are all positive on average, and the Dirichlet policies generally have the higher terminal relative wealth. However, this figure should not be used alone to judge the method. A small return gain can coincide with higher turnover or a low drawdown-feasible rate. The next two slides therefore compare each architecture with its own V2.6 standalone-credit control, rather than treating the highest line as a complete success.

## 7. Gaussian: counterfactual cost is most useful

**Key Points**
- CF cost: relative wealth +4.61% -> +4.97%
- Mean MDD 19.14% -> 18.03%; feasibility 15% -> 30%
- Turnover rises from 1.91% to 7.22% per step

**Speaking Script**

For Gaussian, the counterfactual-cost experiment gives the clearest improvement. Mean relative wealth increases by about zero point three six percentage points. Mean maximum drawdown falls by about one point one one percentage points, and six of twenty test branches satisfy the drawdown target rather than three. This is useful evidence that cost-side branch credit can matter. It is not a free improvement: mean full-L1 turnover rises from about one point nine percent to seven point two percent per step. Counterfactual reward alone is essentially tied with the control, and changing both reward and cost is weaker than changing cost alone. I would therefore retain the cost-only result as the strongest Gaussian finding, without claiming that the risk problem is solved.

## 8. Dirichlet: highest return, only a small gain

**Key Points**
- CF reward has the highest observed return: +8.18%
- Standalone Dirichlet is already +8.06%
- Turnover 14.27% -> 24.00%; win rate 90% -> 85%

**Speaking Script**

Counterfactual reward produces the highest observed mean relative wealth, eight point one eight percent. But the matched standalone control is already at eight point zero six percent. The gain is only around zero point one two percentage points. Meanwhile, turnover rises substantially and the fraction of markets where the model beats the baseline falls from ninety to eighty-five percent. The lower maximum drawdown and higher drawdown-feasible rate are encouraging, but these are small changes over one training seed. The manuscript correctly treats this as an observed leader rather than conclusive superiority. The cost-only and joint variants do not improve return over the control. This makes the result architecture dependent, rather than a universal advantage of counterfactual credit.

## 9. Hard constraints work; soft risk remains open

**Key Points**
- Zero allocation violation for every evaluated simplex policy
- Best V3.2 return-selected drawdown-feasible rate: 45%
- Mean MDD and online drawdown feasibility are different metrics

**Speaking Script**

The strongest structural result is unchanged: all evaluated simplex actions satisfy the allocation constraints. That does not translate into guaranteed drawdown feasibility. Even the best V3.2 return-selected policy satisfies the active drawdown-cost target on only nine of twenty branches. Maximum drawdown summarises the worst peak-to-trough loss. The constraint instead compares current drawdown with a time-varying benchmark budget, so similar maximum drawdown can hide very different cost histories. A branch is feasible when its average actual cost does not exceed its average dynamic alpha, not when every step has zero violation. This distinction is central to the final report and to how we evaluate the role of RCPO.

## 10. Checkpoint selection is part of the limitation

**Key Points**
- Return-best and feasibility-ranked checkpoints answer different questions
- Feasibility-ranked does not guarantee test feasibility
- CF cost Dirichlet: return +6.18% -> +0.92%, feasibility stays 40%

**Speaking Script**

I also reviewed the separately saved feasibility-ranked checkpoints. The rule prioritises validation feasible-branch rate and then validation return. The file name should not be interpreted as a guarantee. On the unseen test markets, the highest feasible rate is still forty-five percent. The cost-only Dirichlet example is particularly revealing: choosing its feasibility-ranked checkpoint reduces test relative wealth from six point one eight percent to zero point nine two percent, while the test feasible rate stays at forty percent. This does not mean that we should select using test results. It means that the validation selection rule is not yet reliably aligned with future risk-return performance, and its name and interpretation need to remain precise.

## 11. What the project demonstrates

**Key Points**
- Established: feasible action construction and reproducible evaluation
- Measured: architecture-dependent credit and turnover trade-offs
- Unresolved: future risk feasibility and training-seed robustness

**Speaking Script**

My overall assessment is that the contribution is methodological rather than a claim of real-market profitability. We have a working separation between action-level constraints and path-level risk preferences, plus a controlled experiment on branch credit. The counterfactual-cost Gaussian result is a meaningful risk-side signal, while the Dirichlet reward result is much more marginal than the highest return number initially suggests. There is no uniform winner across return, risk, feasibility and turnover. Historical versions also used different markets and constraint groups, so a lower raw return in a newer version is not enough to diagnose regression. The matched V2.6 comparison is more informative. The main open question is how to make risk learning and model selection transfer reliably to unseen paths.

## 12. Final report: a complete draft for review

**Key Points**
- Draft written: formulation, CAOSD + RCPO, open-loop credit
- Completed result tables and return / drawdown figures included
- Discussion separates evidence, interpretation and remaining limitations

**Speaking Script**

I have written the LaTeX report under the title Towards Robust Portfolio Reinforcement Learning with Constraint-Aware Simplex Decomposition. It now explains the portfolio process, the two hard constraints, the benchmark-relative current-drawdown cost, the counterfactual variants, and the completed evaluation. The conclusion is deliberately narrower than saying that counterfactual credit solves the problem. Before submission, I would like feedback on whether the contribution should be framed mainly as the hybrid constrained design or as the evaluation of branch credit within that design. I also need a final technical and presentation pass: formula notation, source-to-table consistency, figure readability, bibliography checks, and compilation. The current draft should be treated as ready for discussion, not as already approved or publication ready.

## 13. Discussion and next decisions

**Key Points**
- Is the hybrid design plus controlled credit study a sufficient core contribution?
- Prioritise more seeds and friction sensitivity, or a new risk objective?
- Keep real-market validation and reward correction as future work?

**Speaking Script**

There are three decisions I would like to discuss. First, is the current hybrid design and controlled counterfactual study a sufficient core for the MEng report, with the limitations stated clearly? Second, if there is time for one more experiment, I would prioritise replication across training seeds and transaction-cost sensitivity before another broad hyperparameter sweep. If the priority is strict risk feasibility instead, that may require a different risk objective or model-selection procedure rather than merely increasing lambda. Third, should historical-market validation and reward correction remain explicitly outside the completed experimental scope? My proposed closing message is that CAOSD successfully handles the hard allocation problem, while RCPO risk learning and counterfactual credit produce useful but incomplete improvements.

## Appendix A. Complete Return-Best Comparison

All rows use the same 20 test continuations. MDD and turnover are percentages. Turnover is full-L1 reallocation per step, not cost in basis points. DD feasibility means episode-average actual constraint cost <= episode-average dynamic alpha.

| Policy | Relative wealth | Win rate | Mean MDD | Turnover | DD feasible |
|---|---:|---:|---:|---:|---:|
| V2.6 Gaussian Control (Best Return) | +4.61% | 70% | 19.14% | 1.91% | 15% |
| CF Reward Gaussian v3.2 | +4.61% | 70% | 19.20% | 0.96% | 10% |
| CF Cost Gaussian v3.2 | +4.97% | 75% | 18.03% | 7.22% | 30% |
| CF Reward+Cost Gaussian v3.2 | +4.13% | 65% | 18.66% | 4.25% | 10% |
| V2.6 Dirichlet Control (Best Return) | +8.06% | 90% | 16.73% | 14.27% | 40% |
| CF Reward Dirichlet v3.2 | +8.18% | 85% | 16.52% | 24.00% | 45% |
| CF Cost Dirichlet v3.2 | +6.18% | 85% | 17.00% | 17.14% | 40% |
| CF Reward+Cost Dirichlet v3.2 | +6.49% | 75% | 16.85% | 22.89% | 35% |

The neutral baseline has zero relative wealth by definition and mean MDD 16.70%. Its allocation is feasible but is not arithmetic 1/N. No drawdown-feasibility claim is assigned to this untrained reference.

## Appendix B. Method Details for Questions

- Current drawdown: D_t = 1 - wealth_t / running_peak_t; it can decrease after recovery. It is not running maximum drawdown.
- Budget: b_t = max(0.05, 0.90 * D_baseline,t).
- Actual cost: c_t = max(0, D_t - b_t)^2 / 0.10.
- Dynamic tolerance: alpha_t = (0.05 * b_t)^2 / 0.10. This corresponds to a cost-unit tolerance, not a guaranteed per-step 5% violation allowance.
- One global multiplier update per rollout: lambda = max(0, lambda + lr * (mean(actual_cost) - mean(alpha))). Up/down rates in the completed V3.2 summaries are 0.00075 / 0.01.
- Counterfactual cost is signed actual-minus-counterfactual cost. Lambda still uses actual nonnegative portfolio cost.
- Counterfactual worlds retain other realised branches, including downstream autoregressive actions. Thus the signal is an open-loop intervention heuristic, not an unbiased causal effect.
- Allocation groups: [1,4,6] >= 0.50 and [2,4,7] >= 0.40. The overlap branch has z1=0 here; the overlap still affects the other CAOSD coefficients.
- Gaussian and Dirichlet have matched environment interactions but different optimisation epoch budgets (3 and 4).
- Reward correction and manually injected reward noise are not active in these experiments.

## Appendix C. Draft Review Checklist

- Confirm contribution wording and whether additional experiments are necessary before submission.
- Verify all claims against shared-test JSON/CSV and distinguish validation selection from test reporting.
- Check equations and compile the LaTeX. One source-level typo is visible in the lambda equation: `]_+,qquad` should be reviewed as intended `]_+,\qquad`. It is noted here only; the manuscript was not modified.
- The draft calls the reward improvement uncertain; any interval must remain conditional on one training seed.
- Verify bibliography metadata and figure readability during the final submission pass.
- Keep historical data, multi-seed validation, execution realism, and reward correction clearly scoped as uncompleted extensions.

## Appendix D. Sources and Run Traceability

- [Current LaTeX draft](C:\Users\miaoj\Desktop\RCPO in trading\reports\final_project_report.tex)
- [Return-best evaluation summary](C:\Users\miaoj\Desktop\RCPO in trading\evaluation\section9_simplex_v3.2_policy_comparison\section9_comparison_summary.json)
- [Feasibility-ranked evaluation summary](C:\Users\miaoj\Desktop\RCPO in trading\evaluation\section9_simplex_v3.2_best_feasible_policy_comparison\section9_comparison_summary.json)
- CF Reward Gaussian v3.2: [training summary](C:/Users/miaoj/Desktop/RCPO in trading/runs/simplex_v3.2_cf_reward_gaussian_rcpo_none_20260902_195405/seed_0/training_summary.json), 60000 completed updates; evaluated checkpoint: checkpoint_best_return.pt.
- CF Cost Gaussian v3.2: [training summary](C:/Users/miaoj/Desktop/RCPO in trading/runs/simplex_v3.2_cf_cost_gaussian_rcpo_none_20260902_195411/seed_0/training_summary.json), 60000 completed updates; evaluated checkpoint: checkpoint_best_return.pt.
- CF Reward+Cost Gaussian v3.2: [training summary](C:/Users/miaoj/Desktop/RCPO in trading/runs/simplex_v3.2_cf_reward_cost_gaussian_rcpo_none_20260902_195417/seed_0/training_summary.json), 60000 completed updates; evaluated checkpoint: checkpoint_best_return.pt.
- CF Reward Dirichlet v3.2: [training summary](C:/Users/miaoj/Desktop/RCPO in trading/runs/simplex_v3.2_cf_reward_dirichlet_rcpo_none_20260902_195424/seed_0/training_summary.json), 60000 completed updates; evaluated checkpoint: checkpoint_best_return.pt.
- CF Cost Dirichlet v3.2: [training summary](C:/Users/miaoj/Desktop/RCPO in trading/runs/simplex_v3.2_cf_cost_dirichlet_rcpo_none_20260902_195431/seed_0/training_summary.json), 60000 completed updates; evaluated checkpoint: checkpoint_best_return.pt.
- CF Reward+Cost Dirichlet v3.2: [training summary](C:/Users/miaoj/Desktop/RCPO in trading/runs/simplex_v3.2_cf_reward_cost_dirichlet_rcpo_none_20260902_195436/seed_0/training_summary.json), 60000 completed updates; evaluated checkpoint: checkpoint_best_return.pt.

V2.6 controls use the frozen `checkpoint_best_return_eval_snapshot.pt` copies recorded in the comparison summary, not newly trained controls.

### Figures

![Mean relative wealth on 20 shared test markets](C:/Users/miaoj/Desktop/RCPO in trading/evaluation/section9_simplex_v3.2_policy_comparison/section9_cumulative_return_means_only.png)

![Maximum drawdown on 20 shared test markets](C:/Users/miaoj/Desktop/RCPO in trading/evaluation/section9_simplex_v3.2_policy_comparison/section9_max_drawdown_comparison.png)
