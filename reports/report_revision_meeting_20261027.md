# Report Revision Update and CF Discussion

Professor meeting · 27 October 2026  
Compared with the 19 October outline

**Main message:** I have kept the original research direction and made the formulation and learning design more explicit. The main question for discussion is whether counterfactual credit assignment (CF) should become a central empirical contribution.

## 1. What has changed since 19 October?

The earlier outline already established the general allocation setting, the hard/soft constraint distinction and the branch-credit question. This revision develops those ideas:

- **Clearer structure:** Preliminaries introduces the established methods; Proposed Method explains their integration and the implemented learning choices.
- **Rebuilt formulation:** the section now starts with the allocation system and its dynamics, then defines information and the policy objective.
- **More accurate method description:** branch weighting, likelihood ratios, critic inputs, entropy regularisation and multiplier timing are explicit.
- **More careful interpretation:** the experiments compare training schemes whose effects cannot yet be attributed to the CF signal alone.

**The contribution remains formulation, integration and credit-learning comparison.** CAOSD, PPO and RCPO retain their original attribution.

## 2. The new problem-formulation flow

1. **Allocation system:** what is being allocated, and which actions are admissible?
2. **Portfolio evolution:** how do returns and changes in target weights affect wealth?
3. **Path risk:** how do wealth history and the benchmark determine risk cost and allowance?
4. **Information and objective:** what can the policy observe, and what should it optimise?

This follows the recommended paper's system-first approach: quantities and dynamics come before the formal control problem.

Three details are now clearer:

- Previous weights determine transaction costs; wealth and peaks provide risk memory.
- The actor observes market features and diagnostics, while the full environment state contains additional accounting information.
- Reward is discounted, while the risk requirement uses average cost. The method explains the guiding role of discounted cost advantages.

The transaction model remains a target-weight-change approximation. Detailed accounting assumptions and numerical safeguards are in the appendix.

## 3. Why could CF become a central contribution?

**Research question: Can portfolio-level counterfactual feedback help dependent branches learn their role in the allocation that is actually executed?**

Standalone credit evaluates a branch invested on its own. Its contribution to the final portfolio can differ because:

- The branch may receive only a small allocation mass.
- Its exposure can change the mass assigned to later branches.
- Transaction costs and drawdown depend on the combined portfolio and its history.

CF addresses this mismatch by replacing one branch with a neutral allocation and recomputing the feasible map. Each replacement path maintains its own weights, wealth and peak.

This gives the paper three connected questions:

| Component | Question it addresses |
| --- | --- |
| Simplex / CAOSD | Which allocations can be executed? |
| RCPO-inspired learning | How should the penalty on path risk adapt? |
| CF credit | How should shared outcomes be attributed to branches? |

**CF could have comparable importance in the argument, although these components have different guarantees and evidence.** It could give the study a sharper empirical focus than integration alone.

The potential contribution is its adaptation and evaluation within constrained, autoregressive allocation with path-dependent cost. Counterfactual credit itself has established precedents in [COMA](https://arxiv.org/abs/1705.08926) and [Difference Rewards Policy Gradients](https://arxiv.org/abs/2012.11258).

## 4. What does the current evidence support?

- **Return:** the strongest observed Dirichlet CF result has an inconclusive advantage over its standalone control and higher turnover.
- **Consistency:** rankings depend on the actor and evaluation criterion. Combining reward and cost CF gives no consistent additional benefit.
- **Risk feasibility:** none of the evaluated return-best RCPO policies meets the pooled empirical average-cost target.

**Current defensible conclusion:** the credit schemes produce different observed trade-offs; a reliable advantage from CF itself has not been established.

The main attribution problem is that credit modes also differ in mass weighting, training masks, entropy weighting and critic inputs. These factors need to be separated before assigning an improvement to the difference signal.

The current CF signal is an **open-loop path comparison**. Other realised branches remain fixed, and shadow histories accumulate earlier replacements. It should therefore be presented as a credit heuristic, without importing unbiasedness or convergence guarantees.

**MDD status:** the evaluator now includes initial wealth, but the archived drawdown values, figure and interval still need re-evaluation. Risk rankings remain provisional.

## 5. What would make a stronger CF contribution convincing?

1. **Isolate the credit signal.** Hold comparable learning choices fixed, or ablate mass weighting, regularisation and critic inputs separately.
2. **Add a comparable global-credit baseline.** Test whether branch-specific credit helps beyond giving branches the shared portfolio signal.
3. **Repeat across training seeds.** Shared test paths compare trained policies; they do not establish training stability.
4. **Explain the mechanism and cost.** Examine changes in mapped allocations and credit signals, sensitivity to neutral replacement and shadow history, and additional computation and turnover.

A useful result could identify **when CF helps, what behaviour it changes, and when it adds little value**. That would require controlled evidence, even if the conclusion is conditional rather than uniformly positive.

## 6. Which direction should the paper take?

- **Integration-led paper:** keep hard-feasible allocation and adaptive risk learning as the main story, with CF as a substantial comparative component. This is closest to the present evidence.
- **CF-led empirical paper:** make credit assignment the main research question, with CAOSD and RCPO as established foundations. Organise the experiments around the mismatch between isolated branch performance and contribution to a feasible portfolio.

The CF-led direction remains a proposal; it has not been adopted in the manuscript.

**Questions for this meeting:**

- Is the new system-first formulation clear enough?
- Is CF important enough to become the main empirical question?
- After correcting the archived MDD results, which validation should come first: a controlled credit comparison, a global-credit baseline, or training-seed replication?

---

**Materials:** [19 October whiteboard](https://www.tldraw.com/f/e-e_u5Lc8XwVREAi6XCdv) · [19 October speaking guide](<C:/Users/miaoj/Desktop/RCPO in trading/slides/report_concept_meeting_20261019/meeting_speaking_guide_zh.md>) · [Current manuscript](<C:/Users/miaoj/Desktop/RCPO in trading/reports/final_project_report.tex>) · [Formulation outline](<C:/Users/miaoj/Desktop/RCPO in trading/reports/problem_formulation_rebuild_outline.md>) · [CF contribution option](<C:/Users/miaoj/Desktop/RCPO in trading/reports/counterfactual_contribution_outline.md>)
