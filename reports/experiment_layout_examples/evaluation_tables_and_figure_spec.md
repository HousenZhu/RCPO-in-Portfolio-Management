# 评估图表规格与演示数据

本文件用于指导现有评估代码的更新，保留三张表和一张 Figure 1。所有收益结果统一使用 **relative wealth（相对财富）**；Table 1 额外报告算法自身的实际累计收益率。

**数据声明：本文所有数值、置信区间、可行率和计算开销均为假设的展示样例，不是实验结果，也不预设算法优劣。不得将其写入真实评估结果文件或论文实证结论。** 实际输出必须由原始评估轨迹重新计算。旧的示例 PNG 仅供历史版式参考，后续更新以本文件为准；不再制作其余三张图。

## 1. 比较对象：固定保留八个算法

| 顺序 | 建议 algorithm_id | 论文显示名称 | 对应设置与作用 |
|---|---|---|---|
| 1 | `constrained_neutral` | Constrained-neutral | 固定可行配置；相对财富的统一参照 |
| 2 | `rcpo_allocation_penalty` | RCPO with allocation penalties | 单纯形策略；组暴露以违反成本处理，同时控制路径成本 |
| 3 | `caosd_ppo` | CAOSD + PPO | 结构化可行动作，无自适应路径成本乘子 |
| 4 | `caosd_rcpo_shared` | CAOSD + RCPO (shared global) | 共享全局信用的组合方法 |
| 5 | `caosd_rcpo_standalone` | CAOSD + RCPO (standalone) | 使用各分支独立路径的信用 |
| 6 | `caosd_rcpo_cf_r` | CAOSD + RCPO (CF-R) | 反事实奖励信用 |
| 7 | `caosd_rcpo_cf_c` | CAOSD + RCPO (CF-C) | 反事实成本信用 |
| 8 | `caosd_rcpo_cf_rc` | CAOSD + RCPO (CF-RC) | 反事实奖励与成本信用 |

以上 ID 为建议的输出接口，不表示当前代码已具备全部独立运行配置。正式评估前须逐项确认配置、检查点和信用定义。

- Table 1、Table 3 和 Figure 1 均覆盖这八个算法；Table 2 专门比较后五种信用机制。
- 主比较固定策略分布族；Gaussian 与 Dirichlet 如均需展示，应分别输出，不能直接混合平均。
- RCPO with allocation penalties 应记录组暴露与路径成本各自的定义、预算和乘子。所有方法使用相同的评估成本，不能以各自训练目标替代统一指标。
- 记录 `surrogate_type`。当前 `branch_credit_mode="global"` 对应的联合似然更新不能直接冒充 Table 2 中固定条件分支更新后的共享信用对照；若需要该匹配对照，应先确认是否有对应实现与检查点。

## 2. 指标、单位与统计口径

### 2.1 实际累计收益与相对财富

令 \(W_{s,e,t}\) 为算法在训练种子 \(s\)、评估市场 \(e\) 上的实际财富；\(W^B_{e,t}\) 为同一市场上的 constrained-neutral 基准财富。\(H\) 是评估区间长度。算法自身的实际累计收益率为

\[
R^{\mathrm{abs}}_{s,e}=\frac{W_{s,e,H}}{W_{s,e,0}}-1.
\]

相对财富采用项目当前的“基准财富比减一”口径：

\[
R^{\mathrm{rel}}_{s,e}
=\frac{W_{s,e,H}/W_{s,e,0}}{W^B_{e,H}/W^B_{e,0}}-1.
\]

- Table 1 中“真实财富（实际收益 x%）”采用 **Actual cumulative return (%)**：初始财富为 1、期末为 1.1621 时，显示 16.21%。同时保留原始 `terminal_wealth`，便于核查。
- 财富来自环境实际记录的净增长过程；扣除环境已经实现的费用，不额外虚构滑点、冲击或费用。
- 先逐条配对轨迹计算相对财富，再汇总；不能用算法平均财富除以基准平均财富代替，也不能用两个累计收益率直接相减代替。
- 基准相对自身的相对财富恒为 0；其实际累计收益不必为 0。
- 原始数据保存小数：`0.076` 显示为 `7.60%`。两个相对财富之差 `0.0015` 显示为 `+0.15 pp`，即百分点。

### 2.2 路径成本与可行性

对同一评估轨迹，定义平均成本差

\[
G_{s,e}=\frac{1}{H}\sum_{t=0}^{H-1}
\bigl(c_{s,e,t+1}-\alpha_{s,e,t+1}\bigr),
\]

其中 \(c\) 为论文与实现一致的路径风险成本，\(\alpha\) 为对应的状态相关容忍度。表中以 \(10^{-3}\) 为单位展示：原始值 `-0.00006` 显示为 `-0.06`。路径成本可行率为满足 \(G_{s,e}\leq\varepsilon_C\) 的评估轨迹比例；\(\varepsilon_C\) 是预先确定的数值容差。该指标不代表每个时刻均满足回撤限制。

其余指标：

- **Maximum drawdown (MDD)**：每条轨迹先计算最大回撤，再汇总；历史峰值必须包含初始财富。
- **Turnover**：沿用环境完整配置向量的 L1 换手定义，包含现金坐标；不是除以 2 的单边换手。
- **Allocation feasibility**：分别检查非负性、权重和与组暴露。保存动作违反步数比例和最大违反量，使用统一、预先声明的数值容差。
- 所有算法按同一成本与动作可行性规则评估，包括无风险训练目标的方法和静态基准。

### 2.3 重复实验与不确定性

保留 `training_seed`、`market_id`、`action_seed`，区分训练随机性、市场随机性和随机策略采样。优先先汇总每个训练种子内的评估轨迹，再跨独立训练种子汇总。若同时估计市场不确定性，采用保持方法配对关系的种子与市场重采样，不将共享训练模型的全部轨迹当作独立训练重复。

报告均值与 95% 置信区间，并保存计算方法。静态基准没有训练种子，其区间仅反映评估市场等实际存在的随机性。Table 2 的区间直接从配对差估计，不能由两组区间端点相减得到。验证集用于选取检查点，最终测试集不参与选模。

## 3. Table 1：整体表现

**目的：** 在同一测试协议下展示八种方法的相对财富、实际累计收益和路径成本表现。结构化映射与 RCPO 的已有性质作为背景，不包装成本文重新证明的核心贡献；本表重点交代组合方法与信用机制的实际表现。

**建议英文标题：** Overall out-of-sample performance.

下表为假设样例。仅为使两个收益列可核算，假设此示例中的每条基准轨迹实际收益均为 8%，于是实际收益为 \(1.08(1+R^{\mathrm{rel}})-1\)。真实市场基准收益变化时，必须逐轨迹计算，不能对均值套用此换算。

| Method | Relative wealth (%) [95% CI] | Actual cumulative return (%) | Mean cost gap (×10⁻³) [95% CI] | Path feasibility (%) | MDD (%) | Turnover (%) |
|---|---:|---:|---:|---:|---:|---:|
| Constrained-neutral | 0.00 [0.00, 0.00] | 8.00 | +0.35 [+0.22, +0.48] | 35 | 15.0 | 0.0 |
| RCPO with allocation penalties | 7.80 [7.10, 8.50] | 16.42 | +0.10 [−0.03, +0.23] | 48 | 14.7 | 10.9 |
| CAOSD + PPO | 8.80 [7.90, 9.70] | 17.50 | +0.60 [+0.42, +0.78] | 30 | 16.0 | 11.6 |
| CAOSD + RCPO (shared global) | 7.50 [6.80, 8.20] | 16.10 | −0.04 [−0.13, +0.05] | 60 | 14.2 | 9.8 |
| CAOSD + RCPO (standalone) | 7.60 [6.90, 8.30] | 16.21 | −0.06 [−0.14, +0.02] | 62 | 14.0 | 10.0 |
| CAOSD + RCPO (CF-R) | 8.00 [7.20, 8.80] | 16.64 | +0.02 [−0.08, +0.12] | 55 | 14.3 | 13.0 |
| CAOSD + RCPO (CF-C) | 7.20 [6.50, 7.90] | 15.78 | −0.18 [−0.27, −0.09] | 72 | 13.2 | 11.8 |
| CAOSD + RCPO (CF-RC) | 7.70 [6.90, 8.50] | 16.32 | −0.03 [−0.12, +0.06] | 60 | 13.9 | 12.6 |

样例不刻意安排 CF 全面占优。真实结果应据区间、配对差及风险代价解读，不能仅按单个均值选择“最佳算法”。Actual cumulative return 等列也应计算区间；若版面过宽，可将同一张表拆为收益与风险两个分块，共用算法行与表号。

动作可行性作为核查结果另附 Table 1 的紧凑附表或脚注：报告各算法的违反步数比例及最大违反量；除非真实测量支持，不预填全为零。

**建议输出：** `table_1_main_results.csv`

```text
algorithm_id,display_name,actor_family,credit_mode,surrogate_type,
training_seed_count,market_count,action_repeats,
mean_relative_wealth,ci_low_relative_wealth,ci_high_relative_wealth,
mean_actual_cumulative_return,ci_low_actual_cumulative_return,ci_high_actual_cumulative_return,
mean_terminal_wealth,mean_cost_gap,ci_low_cost_gap,ci_high_cost_gap,
path_feasible_rate,mean_max_drawdown,mean_average_turnover,
allocation_violation_step_rate,max_allocation_violation
```

现有汇总字段可作为接入点：`mean_relative_wealth_vs_constrained_neutral` 对应相对财富，`mean_final_cumulative_return` 对应实际累计收益；接入前仍须检查其基准、净收益与聚合口径是否完全一致。

## 4. Table 2：匹配协议下的信用比较，仅面板 A

**目的：** 在相同分支策略更新设置下，检验信用来源变化对应的相对财富与成本差变化。仅保留五种信用机制，不增加另一面板。

**建议英文标题：** Paired credit-assignment comparisons under a matched update protocol.

固定策略分布、条件分支 PPO surrogate、外部质量权重、熵项、训练预算及 critic 配置；将不可避免的差异明确记录。CF-C 中若还含有奖励缩放，必须披露，否则不能把差异仅归于成本信用。匹配后的 standalone 是本表参照，可能不同于 Table 1 原始训练配方，不能直接相减 Table 1 的均值冒充本表结果。

对相同训练种子、市场与评估重复，先计算

\[
\Delta R^{\mathrm{rel}}_{s,e}
=R^{\mathrm{rel}}_{s,e,\mathrm{method}}
-R^{\mathrm{rel}}_{s,e,\mathrm{standalone}},\qquad
\Delta G_{s,e}=G_{s,e,\mathrm{method}}-G_{s,e,\mathrm{standalone}}.
\]

| Credit assignment | Δ Relative wealth (pp) [95% CI] | Δ Mean cost gap (×10⁻³) [95% CI] |
|---|---:|---:|
| Shared global | −0.10 [−0.40, +0.20] | +0.02 [−0.04, +0.08] |
| Standalone | 0.00 (reference) | 0.00 (reference) |
| CF-R | +0.15 [−0.20, +0.50] | +0.05 [−0.03, +0.13] |
| CF-C | −0.25 [−0.50, 0.00] | −0.08 [−0.14, −0.02] |
| CF-RC | +0.05 [−0.25, +0.35] | +0.02 [−0.06, +0.10] |

上表仍为独立的假设样例，不由 Table 1 推导。收益差越大越好，成本差越小越好；跨零区间不支持确定的改善结论。参照行的零是定义值，不是测得的零方差结果。

**建议输出：** `table_2_matched_credit.csv`

```text
comparison_id,algorithm_id,reference_algorithm_id,actor_family,matched_protocol_id,
paired_training_seed_count,paired_market_count,
mean_delta_relative_wealth,ci_low_delta_relative_wealth,ci_high_delta_relative_wealth,
mean_delta_cost_gap,ci_low_delta_cost_gap,ci_high_delta_cost_gap
```

## 5. Table 3：八个算法的计算代价

**目的：** 交代信用改动的训练代价和部署代价。统一硬件、真实环境交互预算及计时边界；影子路径计算单独计数，不算成新的真实市场交互。

**建议英文标题：** Training and deployment costs.

以下归一化样例以 shared-global 方法为 1；20M 仅为演示预算，不替代实际实验设置。

| Method | Real training transitions (M) | Training time (relative) | Peak training memory (relative) | Decision latency (relative) |
|---|---:|---:|---:|---:|
| Constrained-neutral | 0 | N/A | N/A | 0.03 |
| RCPO with allocation penalties | 20 | 0.95 | 0.96 | 0.72 |
| CAOSD + PPO | 20 | 0.92 | 0.99 | 0.99 |
| CAOSD + RCPO (shared global) | 20 | 1.00 | 1.00 | 1.00 |
| CAOSD + RCPO (standalone) | 20 | 1.10 | 1.03 | 1.01 |
| CAOSD + RCPO (CF-R) | 20 | 1.36 | 1.07 | 1.01 |
| CAOSD + RCPO (CF-C) | 20 | 1.39 | 1.08 | 1.00 |
| CAOSD + RCPO (CF-RC) | 20 | 1.48 | 1.13 | 1.01 |

- 训练时间包含环境交互、影子路径与优化；验证耗时是否计入须一致声明。记录真实交互数、影子转移数与优化步数。
- 决策时延测量实际部署调用，包括策略前向与可行映射；固定 batch size，预热，GPU 计时需同步。若部署需要额外上下文计算，也应包含并说明。
- 静态基准的训练耗时与训练显存标为 N/A，原始存储使用 null；决策时延仍需实测。
- 归一化前保存秒、字节和毫秒等绝对值，并提供重复计时的离散程度。归一化结果不能跨不同硬件直接比较。

**建议输出：** `table_3_compute.csv`

```text
algorithm_id,actor_family,hardware_id,batch_size,
training_real_transition_count,shadow_transition_count,optimizer_step_count,
training_wall_time_seconds,peak_training_memory_bytes,decision_latency_ms,
relative_training_time,relative_peak_memory,relative_decision_latency
```

## 6. Figure 1：覆盖八个算法的学习曲线

**建议英文标题：** Validation relative wealth and path-cost gap during training.

为避免八条曲线全部重叠，采用一张 **2 × 2 子图**，共享方法配色；standalone 在两组中重复出现，作为阅读锚点。

| 子图 | 方法 | 横轴 | 纵轴 |
|---|---|---|---|
| (a) Overall comparison | Constrained-neutral、RCPO with allocation penalties、CAOSD + PPO、Standalone | Real environment transitions (M) | Relative wealth (%) |
| (b) Credit comparison | Shared global、Standalone、CF-R、CF-C、CF-RC | 同上 | Relative wealth (%) |
| (c) Overall comparison | 同 (a) | 同上 | Mean cost gap (×10⁻³) |
| (d) Credit comparison | 同 (b) | 同上 | Mean cost gap (×10⁻³) |

这里的相对财富是每个训练检查点在固定验证市场上测得的**期末相对财富**，不是一个评估 episode 内随交易日期变化的累计财富曲线。两行分别保留收益与路径成本信息，不能只用收益曲线宣称信用机制有效。

### 6.1 收益曲线的演示数据

下表数值单位为百分比；0、5M、10M、20M 均为假设展示点。

| Method | 0 | 5M | 10M | 20M |
|---|---:|---:|---:|---:|
| Constrained-neutral | 0.0 | 0.0 | 0.0 | 0.0 |
| RCPO with allocation penalties | 0.0 | 4.0 | 6.5 | 7.8 |
| CAOSD + PPO | 0.0 | 5.0 | 7.8 | 8.8 |
| CAOSD + RCPO (shared global) | 0.0 | 4.2 | 6.6 | 7.5 |
| CAOSD + RCPO (standalone) | 0.0 | 4.4 | 6.8 | 7.6 |
| CAOSD + RCPO (CF-R) | 0.0 | 5.0 | 7.3 | 8.0 |
| CAOSD + RCPO (CF-C) | 0.0 | 4.2 | 6.5 | 7.2 |
| CAOSD + RCPO (CF-RC) | 0.0 | 4.4 | 6.9 | 7.7 |

除基准外，真实初始策略的相对财富不必为零。本表共用零起点只是版式占位；正式绘图必须实测初始策略，不能人为平移曲线。

### 6.2 路径成本曲线的演示数据

数值单位为 ×10⁻³；与收益曲线使用相同检查点。

| Method | 0 | 5M | 10M | 20M |
|---|---:|---:|---:|---:|
| Constrained-neutral | +0.35 | +0.35 | +0.35 | +0.35 |
| RCPO with allocation penalties | +0.35 | +0.22 | +0.14 | +0.10 |
| CAOSD + PPO | +0.35 | +0.48 | +0.55 | +0.60 |
| CAOSD + RCPO (shared global) | +0.35 | +0.08 | −0.01 | −0.04 |
| CAOSD + RCPO (standalone) | +0.35 | +0.06 | −0.03 | −0.06 |
| CAOSD + RCPO (CF-R) | +0.35 | +0.10 | +0.04 | +0.02 |
| CAOSD + RCPO (CF-C) | +0.35 | −0.04 | −0.13 | −0.18 |
| CAOSD + RCPO (CF-RC) | +0.35 | +0.09 | 0.00 | −0.03 |

### 6.3 绘图规则与输出

- 固定验证集，检查点使用相同评估协议；曲线显示跨独立训练种子的均值及 95% 区间，区间估计方法与表格一致。
- 基准为单独计算的静态参考线，不伪造其训练检查点；其成本差一般不为零。若绘制市场不确定性带，应说明其不确定性来源与训练方法不同。
- 两个收益子图使用相同纵轴范围；两个成本子图也使用相同纵轴范围，并绘制成本零线。
- 同一算法在所有子图中配色与线型一致；不依赖颜色区分。标注有效种子数量。
- 保存原始检查点值。缺失检查点保留缺失标记；用于显示的平滑或插值不得写回原始结果，也不用于统计检验。
- Table 1 使用最终测试集，Figure 1 使用验证集；样例终值相同仅方便阅读，真实输出不要求相同。

**建议原始输出：** `figure_1_checkpoint_metrics.csv`

```text
algorithm_id,actor_family,training_seed,checkpoint_id,
training_real_transition_count,validation_market_set_id,
mean_relative_wealth,mean_cost_gap
```

另输出跨种子汇总 CSV，保存每个检查点的有效种子数、两个指标的均值与区间上下界。静态基准另存记录。最终图片保存为 `figure_1_credit_learning.png` 与矢量版 `figure_1_credit_learning.pdf`；本轮仅制定规格，不生成新的结果图。

## 7. 原始记录与验收

所有表格优先从逐轨迹记录生成，避免不同脚本重复实现不同口径。建议 `evaluation_episodes.csv` 字段：

```text
algorithm_id,actor_family,protocol_id,surrogate_type,
training_seed,checkpoint_id,split,market_id,action_seed,episode_length,
initial_wealth,terminal_wealth,baseline_initial_wealth,baseline_terminal_wealth,
actual_cumulative_return,relative_wealth,
mean_cost,mean_cost_allowance,cost_gap,path_feasible,
max_drawdown,average_turnover,
allocation_violation_step_count,max_allocation_violation
```

另存逐步净增长、算法与基准财富、成本、容忍度及换手序列，便于复算。元数据记录代码版本、检查点、市场列表、策略解码方式（随机采样或确定性）、数值保护、容差、置信区间方法与硬件；不同解码方式分别汇总。

示例元数据模板如下；null 为待填项，正式运行前必须填写：

```json
{
  "illustrative": true,
  "relative_wealth_baseline": "constrained_neutral",
  "policy_decoding": "stochastic",
  "wealth_source": "environment_recorded_growth",
  "confidence_level": 0.95,
  "cost_tolerance": null,
  "allocation_tolerance": null,
  "code_revision": null,
  "hardware": null
}
```

真实结果应使用 `illustrative: false`，并保留原始数据来源。验收时逐项检查：

1. Table 1、Table 3、Figure 1 共覆盖八个算法；Table 2 仅包含五种匹配信用机制。
2. 全部收益比较使用相对财富；Table 1 另外报告实际累计收益率。
3. 基准逐轨迹相对财富为零；实际累计收益与净财富递推一致。
4. 先求逐轨迹财富比再汇总，没有混用平均财富比、超额收益或年化收益。
5. Table 2 使用真实配对记录，收益差以百分点展示。
6. 所有方法使用相同路径成本和动作可行性评估；数值容差已记录。
7. 最大回撤包含初始财富，换手口径与环境一致。
8. 静态方法的训练开销为 N/A，推理开销仍实测。
9. 区间来自真实重复实验，训练种子、市场和策略采样的统计层级清楚。
10. 示例数字和旧示例图未被作为实验数据读入；所有最终表图均可从原始记录复算。
