# V3.3-V3.4 Generalization Study

**Meeting date:** 3 October 2026<br>
**Student:** Housen Zhu<br>
**Supervisor:** Professor A. Bereyhi<br>
**Project:** Constraint-Aware Simplex Decomposition for Robust Portfolio Reinforcement Learning<br>
**Google Slides:** [V3.3-V3.4 Generalization Study - revised presentation](https://docs.google.com/presentation/d/1HaLouokPt6Tu-VxkxjqM3ccCwCsSvQeKTCRfL6rI9u0/edit?usp=drivesdk)

---

## 1. V3.3-V3.4 Generalization Study

### Key Words

generalization | replicated evidence | constrained-neutral baseline | return-risk trade-off

### Explanation

This meeting examines whether the improvements observed during training and validation remain reliable on independent future markets. V3.3 explored market diversity, path length, policy architecture, and branch credit. V3.4 then converted the most important questions into controlled three-seed comparisons.

The main result is not a single best configuration. The evidence shows that market count, optimization epochs, counterfactual reward, checkpoint selection, observation window, and drawdown margin interact. Several settings improve one metric while weakening another.

### Speaking Script

Today I will focus on generalization rather than the highest validation score. In V3.3 I explored several changes to the training market and policy. Those results suggested useful directions, but most comparisons used one seed and changed more than one factor. V3.4 therefore used paired experiments and three training seeds for the central questions. I then evaluated every policy on the same 100 independent one-year markets. This gives us stronger evidence about which effects repeat and which effects remain uncertain.

### Figure Source

Cover slide. No chart required.

---

## 2. Progress Since The Last Meeting

### Key Words

V3.3 exploration | 22 new V3.4 seed-runs | 100-market holdout | paired evaluation

### Explanation

Since the last meeting, the project completed:

- Eleven V3.3 exploratory configurations.
- Twenty-two new V3.4 training runs, combined with two reused V3.3 controls.
- Replicated V3.4 conditions with seeds 0, 1, and 2.
- A common holdout evaluation using five independent anchors and 20 continuations per anchor.
- Separate analysis of best-return and robust-feasible-selected checkpoints.

### Speaking Script

The main progress is the scale and structure of the evidence. I completed 22 new V3.4 seed-runs and combined them with two exact V3.3 controls. The central V3.4 conditions now have three independent training seeds. I also replaced the earlier small future-market comparison with five independent anchor histories and 20 future branches from each anchor. Therefore, each checkpoint is tested on 100 one-year future markets. The evaluation cache also lets me compare policies on exactly the same market paths.

### Figure Source

Use a compact experiment timeline derived from the evaluation manifest.

---

## 3. Why V3.4 Was Needed

### Key Words

single-seed uncertainty | validation optimism | confounded comparisons | paired controls

### Explanation

V3.3 identified promising patterns, but it could not establish stable effects because many conditions had only one training seed. Some comparisons also changed market count, path length, or optimization settings together. Fixed validation branches could favor a checkpoint without predicting performance on new anchors.

V3.4 keeps the environment, constraints, episode length, update count, learning-rate schedule, and core RCPO logic fixed. It changes one experimental factor at a time wherever possible.

### Speaking Script

V3.3 was useful for discovering hypotheses, but it was not strong enough for final conclusions. A result from one training seed can reflect initialization or one training-market sample. Also, when two configurations change several settings, I cannot attribute the difference to one factor. The final concern is checkpoint selection. A model may look strong on the fixed validation branches but lose part of that advantage on new anchors. V3.4 addresses these issues with repeated seeds, paired market paths, and a separate holdout set.

### Figure Source

`evaluation/v3.3_v3.4_professor_meeting/diagnostics_03_validation_to_holdout_gap.png`

---

## 4. Replicated Experiment Design

### Key Words

2 or 8 markets | 3 or 4 epochs | standalone credit | counterfactual reward | window | margin

### Explanation

The replicated factorial compares Gaussian RCPO standalone training under four conditions:

| Training markets | PPO epochs | Training seeds |
|---:|---:|---:|
| 2 x 5,040 steps | 3 | 0, 1, 2 |
| 2 x 5,040 steps | 4 | 0, 1, 2 |
| 8 x 5,040 steps | 3 | 0, 1, 2 |
| 8 x 5,040 steps | 4 | 0, 1, 2 |

Counterfactual reward is compared with standalone reward under the 2-market setting at 3 and 4 epochs. Observation windows of 20, 30, 40, and 60 steps and drawdown margins of 0.90, 0.85, and 0.80 are single-seed exploratory sweeps.

### Speaking Script

The central V3.4 design is a two-by-two factorial experiment. I compare two versus eight training markets and three versus four optimization epochs. Each of the four conditions uses the same three seeds. I also compare standalone branch reward against counterfactual branch reward at both epoch settings. Finally, I run smaller exploratory sweeps for the observation window and drawdown margin. These last two sweeps have one seed, so I treat them as hypotheses rather than stable recommendations.

### Figure Source

Use a native 2-by-2 experiment matrix and short side labels for the credit, window, and margin studies.

---

## 5. Independent Evaluation Protocol

### Key Words

five anchors | 20 branches per anchor | 252 steps | common risk rule | two checkpoint objectives

### Explanation

Every policy is evaluated deterministically on five independent anchor histories. Each anchor generates 20 future 252-step continuations, producing 100 matched future markets per checkpoint.

- Return is measured as relative terminal wealth against the constrained-neutral CAOSD baseline.
- Risk uses a common drawdown rule with margin 0.90, floor 0.05, cost scale 0.10, and alpha-budget ratio 0.05.
- `checkpoint_best_return.pt` maximizes validation return.
- `checkpoint_best_feasible.pt` is the robust-feasible-selected checkpoint. It ranks validation feasible-branch rate first and return second.
- Robust-feasible-selected describes the selection rule. It does not guarantee feasibility on new holdout markets.

### Speaking Script

The evaluation uses a shared set of future markets. For each of five anchor histories, I generate 20 independent one-year continuations. Every policy sees exactly the same risky-asset returns. Return is reported relative to the constrained-neutral CAOSD baseline, which already satisfies the hard allocation constraints. For risk, I recompute every policy under one common drawdown rule so that different training margins remain comparable. I evaluate the return-selected and robust-feasible-selected checkpoints separately. The robust label refers to validation selection and is not a safety guarantee on unseen markets.

### Figure Source

Use a native protocol diagram plus a small callout for the two checkpoint types.

---

## 6. V3.3 Exploratory Findings

### Key Words

longer is not monotonic | market diversity changes risk | single-seed evidence

### Evidence

- Gaussian RCPO with `2 x 5,040` steps: **+8.13%** holdout relative wealth.
- Gaussian RCPO with `2 x 10,080` steps: **+9.17%**.
- Gaussian RCPO with `2 x 20,160` steps: **+7.74%**.
- Gaussian RCPO with `8 x 10,080` steps: **+8.07%**, **42%** common feasibility, and **14.17%** mean maximum drawdown.
- Counterfactual-reward Dirichlet reached **+9.72%** return but only **29%** common feasibility.

### Explanation

Extending two training paths from 5,040 to 10,080 steps improved return, but another doubling did not. The 8-market, 10,080-step condition gave up some return while improving the risk metrics. These are useful design signals, but they remain single-seed evidence.

### Speaking Script

V3.3 first tested the professor's suggestion to use fewer but longer markets. With two markets, moving from 5,040 to 10,080 steps improved holdout return from 8.13 to 9.17 percent. Extending again to 20,160 steps reduced return to 7.74 percent. Therefore, longer training histories did not provide a monotonic benefit. The eight-market, 10,080-step condition produced a more favorable risk profile, with 42 percent common feasibility and 14.17 percent mean maximum drawdown. However, these are one-seed findings, so I use them to motivate V3.4 rather than rank final models.

### Figure Sources

- `evaluation/v3.3_v3.4_professor_meeting/v33_01_market_design_return.png`
- `evaluation/v3.3_v3.4_professor_meeting/v33_02_market_design_risk.png`

---

## 7. V3.4 Factorial Results

### Key Words

three seeds | condition means | return dispersion | no universal winner

### Evidence

| Condition | Mean relative wealth | Common feasibility | Mean max drawdown |
|---|---:|---:|---:|
| 2 markets, 3 epochs | +7.66% | 27.7% | 15.53% |
| 2 markets, 4 epochs | +8.03% | 30.7% | 15.09% |
| 8 markets, 3 epochs | +8.06% | 27.7% | 15.50% |
| 8 markets, 4 epochs | +7.01% | 33.0% | 14.95% |

### Explanation

The condition means are close relative to their seed spread. The highest mean return appears under 8 markets and 3 epochs, but the 2-market and 4-epoch condition is nearly identical. The 8-market and 4-epoch condition has lower return and slightly stronger common feasibility.

### Speaking Script

The three-seed factorial does not identify one setting that dominates all metrics. Eight markets with three epochs produces the highest mean return at 8.06 percent, while two markets with four epochs is very close at 8.03 percent. Eight markets with four epochs has the lowest return at 7.01 percent, but its common feasibility is slightly higher. The seed points matter because the condition differences are often smaller than the spread between seeds. I therefore focus on paired effects rather than selecting the largest mean alone.

### Figure Source

`evaluation/v3.3_v3.4_professor_meeting/v34_01_factorial_best_return.png`

---

## 8. Paired Market And Epoch Effects

### Key Words

interaction | paired seeds | conditional effects | fourth epoch not reliable

### Evidence

- `2 to 8 markets` at 3 epochs: **+0.40 percentage points**, positive for 3 of 3 seeds.
- `2 to 8 markets` at 4 epochs: **-1.02 percentage points**, positive for 1 of 3 seeds.
- `3 to 4 epochs` with 2 markets: **+0.38 percentage points**, positive for 1 of 3 seeds.
- `3 to 4 epochs` with 8 markets: **-1.04 percentage points**, positive for 0 of 3 seeds.

### Explanation

Market count and optimizer epochs interact. More markets help consistently under the 3-epoch setting, but the magnitude is small. A fourth epoch does not provide a stable benefit and is consistently harmful with eight markets.

### Speaking Script

The paired analysis changes the interpretation. At three epochs, increasing the market count from two to eight improves return for all three seeds, but the average gain is only 0.40 percentage points. At four epochs, the same market-count change reduces return by 1.02 percentage points. The fourth epoch gives a small average gain with two markets, but only one seed improves. With eight markets, all three seeds become worse. This suggests an interaction between data diversity and optimization intensity. A fourth epoch is not a generally better setting.

### Figure Source

`evaluation/v3.3_v3.4_professor_meeting/v34_04_factorial_paired_effects.png`

---

## 9. Counterfactual Reward

### Key Words

branch credit | no replicated return gain | risk-return exchange | seed dependence

### Evidence

At 3 epochs, counterfactual reward changes:

- Mean relative wealth by **-1.20 percentage points**.
- Common feasibility by **+6.7 percentage points**.
- Mean maximum drawdown by **-0.68 percentage points**.
- Return is lower for all 3 seeds.

At 4 epochs, the return effect is **-0.33 percentage points**, again lower for all 3 seeds. The risk effects are mixed across seeds.

### Explanation

The current open-loop counterfactual reward does not improve return reliably. At three epochs it shifts the policy toward lower drawdown and higher common feasibility, but it sacrifices return. The treatment therefore acts more like a risk-oriented credit transformation than a general return improvement.

### Speaking Script

The counterfactual reward experiment tests whether branch-specific marginal reward improves credit assignment. It does not produce a replicated return gain. At three epochs, all three seeds lose return and the mean reduction is 1.20 percentage points. At the same time, common feasibility improves by 6.7 percentage points and maximum drawdown decreases by 0.68 percentage points. At four epochs, the return cost is smaller but still negative for all seeds, while risk changes are inconsistent. I interpret this as a return-risk trade-off, not evidence that the present counterfactual method learns a better return policy.

### Figure Sources

- `evaluation/v3.3_v3.4_professor_meeting/v34_05_credit_return.png`
- `evaluation/v3.3_v3.4_professor_meeting/v34_06_credit_risk.png`

---

## 10. Checkpoint Return-Risk Trade-off

### Key Words

selection objective | validation feasibility | holdout risk | no safety guarantee

### Evidence

Across the six replicated V3.4 conditions:

| Checkpoint selection | Relative wealth | Common feasibility | Mean max drawdown |
|---|---:|---:|---:|
| Best return | +7.49% | 30.7% | 15.17% |
| Robust-feasible-selected | +3.58% | 45.4% | 14.35% |

### Explanation

Risk-oriented checkpoint selection transfers partially to independent markets. It improves common feasibility by 14.7 percentage points and lowers drawdown, but it gives up 3.91 percentage points of relative wealth. Holdout feasibility remains below 50%, so the checkpoint cannot be described as safely feasible.

### Speaking Script

Checkpoint selection creates the clearest return-risk trade-off in the study. Return-selected checkpoints average 7.49 percent relative wealth and 30.7 percent common feasibility. Robust-feasible-selected checkpoints average 3.58 percent relative wealth and 45.4 percent feasibility, with lower mean maximum drawdown. This shows that validation-based risk selection has some transfer value. However, fewer than half of the holdout branches satisfy the common rule, so I should not call these checkpoints feasible without qualification. The correct label is robust-feasible-selected.

### Figure Source

`evaluation/v3.3_v3.4_professor_meeting/overview_02_checkpoint_tradeoff.png`

---

## 11. Window And Margin Studies

### Key Words

exploratory sweeps | longer context | stricter training rule | common-rule evaluation

### Evidence

Observation windows:

- Window 20: **+8.26%** return, **15.17%** drawdown, **31%** common feasibility.
- Window 60: **+8.85%** return, **16.15%** drawdown, **24%** common feasibility.

Drawdown margins relative to the matched 0.90 control:

- Margin 0.85: **-4.62 percentage points** return and **-10 percentage points** common feasibility.
- Margin 0.80: **-1.58 percentage points** return and **-2 percentage points** common feasibility.

### Explanation

The 60-step window improves mean return but worsens drawdown, tail behavior, and feasibility. Tightening the training margin does not improve risk under the common 0.90 evaluation rule. These are single-seed findings and require replication.

### Speaking Script

The observation-window sweep suggests that more history can increase return, but the policy also becomes riskier. The 60-step model reaches 8.85 percent relative wealth, while maximum drawdown rises to 16.15 percent and common feasibility falls to 24 percent. The margin sweep gives another caution. Training with stricter margins of 0.85 or 0.80 does not improve feasibility when every model is evaluated under the same 0.90 rule. The stricter settings also reduce return. Because both sweeps use one seed, I treat these patterns as hypotheses for a later replicated study.

### Figure Sources

- `evaluation/v3.3_v3.4_professor_meeting/v34_08_window_sweep.png`
- `evaluation/v3.3_v3.4_professor_meeting/v34_09_margin_sweep.png`

---

## 12. Validation Optimism And Cross-Version Evidence

### Key Words

selection optimism | independent anchors | cross-version comparison | persistent generalization gap

### Evidence

Mean holdout minus selected-validation relative wealth:

- V2.6: **-1.72 percentage points**.
- V3.2: **-1.99 percentage points**.
- V3.3: **-1.83 percentage points**.
- V3.4: **-3.50 percentage points**.

### Explanation

The independent holdout remains weaker than the validation result in every version. V3.4 has the largest average gap, despite its more controlled training experiments. This suggests that checkpoint selection against a fixed validation set remains a central weakness.

### Speaking Script

The broader comparison shows that the validation-to-holdout gap persists across versions. V2.6, V3.2, and V3.3 lose about 1.7 to 2.0 percentage points when moving from selected validation performance to the independent holdout. V3.4 loses 3.50 percentage points on average. The more rigorous V3.4 experiment therefore improves our confidence in the diagnosis, but it does not solve checkpoint-selection optimism. The next design should consider rotating validation anchors, a larger checkpoint-selection set, or a selection criterion that explicitly balances return and risk across independent branches.

### Figure Sources

- `evaluation/v3.3_v3.4_professor_meeting/diagnostics_03_validation_to_holdout_gap.png`
- `evaluation/v3.3_v3.4_professor_meeting/overview_01_cross_version_return_risk.png`
- `evaluation/v3.3_v3.4_professor_meeting/overview_03_selected_relative_wealth_paths.png`

---

## 13. Current Conclusions And Discussion

### Key Words

conditional effects | checkpoint optimism | return-risk objective | next experiment

### Conclusions Supported By The Data

1. More training markets provide a small and consistent return benefit at 3 epochs, but not at 4 epochs.
2. A fourth optimization epoch does not provide a stable benefit and is harmful under the 8-market condition.
3. The current counterfactual reward does not improve return across seeds. At 3 epochs it exchanges return for better drawdown and feasibility.
4. Robust-feasible-selected checkpoints improve holdout risk metrics, but they do not guarantee feasibility.
5. Longer observation windows and stricter drawdown margins do not yet provide a reliable improvement.
6. Validation optimism remains an important unresolved issue.

### Claims Not Yet Supported

- That eight training markets are universally better than two.
- That four epochs are preferable to three.
- That counterfactual reward improves the final policy overall.
- That a stricter training margin creates a safer holdout policy.
- That the robust-feasible-selected checkpoint is feasible on unseen markets.

### Discussion Questions

1. Should the next study prioritize rotating validation anchors or a joint return-risk checkpoint score?
2. Is the counterfactual reward result useful as a risk-oriented method, or should the branch-credit design be changed again?
3. Should the main configuration use eight markets and three epochs because its market-count effect is replicated, despite the small magnitude?
4. Should the next robustness test replicate the 20-step and 60-step windows, or move toward real-market data first?

### Speaking Script

My current conclusion is that the main algorithm can beat the constrained-neutral baseline on average, but the training choices have conditional effects and the risk constraint does not transfer strongly enough. The most repeatable market-count result appears at three epochs, where all three seeds improve when moving from two to eight markets, although the gain is small. Four epochs do not provide a stable advantage. Counterfactual reward reduces return in every paired seed, but it may be useful as a risk-oriented credit mechanism at three epochs. The largest remaining methodological issue is checkpoint selection. I would like to discuss whether the next experiment should first improve the validation and selection protocol, refine the counterfactual credit design, or begin a carefully controlled real-market study.

### Figure Source

Use a closing summary with three evidence-backed findings and the four discussion questions.

---

## Evidence Files

- `evaluation/v3.3_v3.4_professor_meeting/seed_summary.csv`
- `evaluation/v3.3_v3.4_professor_meeting/paired_effects.csv`
- `evaluation/v3.3_v3.4_professor_meeting/comparison_summary.json`
- `evaluation/v3.3_v3.4_professor_meeting/evaluation_manifest.csv`

## Interpretation Notes

- V3.4 replicated conditions use three independent training seeds and provide the main evidence.
- V3.3, observation-window, and drawdown-margin results use one training seed and remain exploratory.
- The 100 future markets are paired evaluation paths, not 100 independent training replications.
- Hard allocation feasibility comes from CAOSD. Common feasibility in this report refers to the common relative drawdown rule.
- Robust-feasible-selected identifies how a checkpoint was chosen on validation data. It does not certify feasibility on holdout markets.
