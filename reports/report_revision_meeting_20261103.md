# Weekly Research Update: Counterfactual Credit in Constrained Allocation

3 November 2026

**Core question: How should coupled allocation branches receive credit for the outcomes they jointly produce?**

## 1. Research focus

CF is now a core research focus within general constrained allocation RL, with portfolio allocation as the empirical instance.

- **CAOSD constructs feasible actions:** the simplex and two group limits are satisfied by the action map.
- **RCPO-inspired learning adapts risk pressure:** a global multiplier responds to actual path cost.
- **CF assigns branch feedback:** feasible replacement paths provide a reference for evaluating each branch.

CAOSD, RCPO and counterfactual credit retain their established attribution. The contribution concerns credit assignment when allocation branches and trajectory outcomes are coupled.

## 2. Why standalone credit is insufficient

**A branch's standalone performance can differ from its contribution to the executed allocation.**

- It receives only part of the resource in the combined action.
- Its choice can change the mass assigned to other branches.
- Adjustment costs and risk depend on the combined action and its history.

This motivates evaluating a branch through its effect within the feasible combination.

## 3. CF construction and analysis

**Replace one branch → recompute the feasible map → update its shadow path → compare outcomes.**

The replacement is uniform on the branch's support. Other realised branch values remain fixed. Each shadow account uses the same market returns but maintains its own allocation, turnover, wealth and drawdown history.

The manuscript identifies two sources of credit:

- **Mapping effects:** replacing a branch can redistribute mass between later branches.
- **History effects:** earlier replacements can produce different costs even when current allocations coincide.

The revised formulation specifies the memory needed to update both actual and shadow outcomes consistently. The comparison is open-loop because downstream samples remain fixed and future actions come from the actual actor; it carries no general unbiasedness guarantee.

## 4. What would establish its value?

**Does CF provide useful and reproducible learning feedback under comparable conditions?**

Existing comparisons also differ in branch weighting, entropy regularisation and critic inputs. The next validation should therefore:

1. Isolate credit under matched branch objectives and a clear weighting and information protocol.
2. Repeat training with independent seeds.
3. Diagnose mapping and history effects, alongside computational overhead.

The intended outcome is an explanation of **when CF helps, which behaviour it changes and when its added complexity is worthwhile**.

[Current manuscript](<C:/Users/miaoj/Desktop/RCPO in trading/reports/final_project_report.tex>)
