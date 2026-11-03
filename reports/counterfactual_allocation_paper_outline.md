# Constrained Allocation RL：以反事实分支信用为核心研究主线的论文大纲

日期：2026-11-02  
模式：academic-research-suite / ars-outline  
状态：**待作者确认，仅为结构与证据规划；未修改论文、代码或实验，未启动新的训练或评估。**

本方案基于当前英文 LaTeX、V2.6/V3.2 归档结果及对应实现。此前关于 CF 的备选文件作为历史讨论保留，本文件是本轮面向投稿的完整结构提案。不自动纳入 V3.3/V3.4，也不将待做的实验或分析描述为已完成成果。

## 1. 核心研究问题与贡献定位

### 一句话核心研究问题

**当配置分支通过硬可行映射和自回归采样相互耦合，且收益与风险依赖持续的轨迹状态时，应如何定义并验证分支反事实信用，使其成为可解释、可复现且具有明确适用边界的 constrained allocation RL 学习机制？**

英文研究问题：

> How can branch-level counterfactual credit be defined and evaluated for structurally feasible allocation policies when branch decisions are autoregressively coupled and rewards and costs depend on persistent trajectory state?

### 一句话贡献定位

**以既有 CAOSD 和 RCPO-inspired learning 为动作与风险学习基础，将“可行映射内重算、独立路径记忆和分支信用的受控验证”作为论文主要增量，研究 CF 在什么条件下有效、为何有效以及何时失效；投资组合提供具体实例，而不是一般性结论的唯一依据。**

这是拟形成的投稿贡献定位，不是对当前证据已经充足的判断。

### 建议的英文工作标题

**Counterfactual Credit in Constrained Allocation Reinforcement Learning with Path-Dependent Costs**

标题将 CF 放到研究中心，同时保留配置约束与路径成本。暂不在标题中使用 improved、optimal、safe 或 guaranteed risk control。标题、摘要和结论只能在证据完成后按实际发现定稿。

### 三条主线的不同角色

| 主线 | 论文回答的问题 | 原创性与证据边界 |
| --- | --- | --- |
| Hard allocation feasibility | 每个执行动作如何满足单纯形和两组暴露约束？ | 采用并归属已有 CAOSD 构造；已有可行性证明是基础，不重新包装为原创结果 |
| Adaptive path-risk learning | 如何用实际成本残差调整策略的风险压力？ | 采用 RCPO-inspired 对偶调节；当前实现没有可靠满足平均风险目标的证据 |
| Counterfactual branch credit | 如何评价依赖分支对可行组合及其路径结果的贡献？ | 已实现开环中性替换与影子路径；形式分析、作用归因及跨种子验证是需要补强的主要研究内容 |

三者共同构成论文，但不声称具有同样的理论保证。CF 获得独立的问题动机、方法定义、分析和实验问题，不能只在末尾增加一个 CF 变体段落。

### 拟形成的三项贡献及其完成状态

| 拟贡献 | 当前已具备 | 投稿前需要补齐 |
| --- | --- | --- |
| C1：明确分离硬动作可行性与期望路径成本的配置控制框架 | N 维动作、两组映射、配置记忆、风险记忆及平均成本目标 | 精简统一表述，保留组件归属及实际支持条件 |
| C2：针对耦合配置分支的路径一致开环 CF 信用定义与性质分析 | 中性替换后重新映射；actual/shadow 各自记账；CF-R、CF-C、CF-RC | 映射耦合、历史状态影响、基线性质的正式分析；不能直接继承 COMA 的结论 |
| C3：界定 CF 作用条件的可复现实证研究 | 单训练种子、20 条共享合成市场延续下的探索性比较 | 修正后重评、受控消融、独立训练种子及外部验证；不能预写“CF 显著优越” |

## 2. 发表定位与证据门槛

### 目标论文类型

采用**方法与分析驱动的应用机器学习期刊论文**结构：一般 constrained allocation RL 为问题主线，投资组合为主要应用实例。主要读者应能理解信用机制及其边界，即使不关心金融收益排名。

“次顶级”没有统一的学科或期刊定义。本轮以有竞争力的应用 AI／智能决策期刊为设计目标，不使用影响因子或分区推断录用机会，也不承诺通过结构调整即可达到投稿要求。

尚未确认具体 venue、article type 或其评审标准，因此：

**criteria_binding_unavailable**

以下只是期刊方向参照，不构成已完成的 venue-fit 判断：

- **Expert Systems with Applications**：出版方介绍涵盖智能系统的设计、测试与应用，也包括金融和风险评估；同时强调清楚说明相对已有方法的实质适配。若以 portfolio 为主要证据，应优先补充可信的历史数据检验、计算开销及方法消融。[出版方介绍](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)
- **Engineering Applications of Artificial Intelligence**：出版方强调 AI 的实际工程应用、新的 AI 方面以及可复现的数据验证。如果选择这一路线，应增加真实工程含义的资源配置任务；仅将股票改名为服务器不构成工程验证。[出版方介绍](https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976)

上述适配建议是根据公开 scope 作出的规划推断。正式选刊前还需确认具体期刊及其届时的作者指南。本轮没有完成投稿格式、政策或分区审计。

### 最值得形成的研究增量

论文潜力主要来自：**指出孤立分支表现与可行组合贡献之间的结构差异，清楚解释路径反事实的评价对象，再用受控实验检验其实际价值。**

单纯罗列 CAOSD、PPO、RCPO 和 CF 不能充分建立方法增量。已有 difference rewards、COMA 和资源分配信用文献必须进入定位；本轮定向检索不足以支持“首次”或“没有前人研究”的声明。

## 3. 子问题与全文论证顺序

| 子问题 | 核心内容 | 主要承载章节 |
| --- | --- | --- |
| RQ1：配置几何如何影响信用？ | 中性替换会不会改变其他分支的组合质量，为什么需要重新映射？ | 2、3、4.1–4.3 |
| RQ2：路径记忆如何影响信用？ | 当前动作差异与此前替换积累的状态差异分别意味着什么？ | 2、4.2–4.3、5 |
| RQ3：CF 本身是否带来可重复的价值？ | 如何区分信用信号、质量权重、熵正则、critic 输入和训练随机性的影响？ | 4.4、5 |
| RQ4：适用范围与成本是什么？ | 哪些配置环境可计算该比较，何时收益不足以补偿额外开销？ | 4.5、5、6 |

全文逻辑：

**配置约束与路径成本 → 可行策略仍面临信用分配问题 → 定义可行且有历史记忆的 CF → 分析比较对象与边界 → 用受控实验检验作用 → 给出条件性的适用结论。**

## 4. 章节设计：中心论点、内容与所需证据

保留 Introduction、Problem Formulation、Preliminaries、Proposed Method、Experimental Evaluation、Discussion、Conclusion 的大章节结构。建议正文约 6,450 个英文词，作为软预算而非篇幅硬限制；摘要、参考文献与附录另计。

### Abstract（约 200 词，不计入正文预算）

**中心论点：** 给出耦合分支与路径信用问题、采用的构造、实际验证范围和主要发现。

内容按“问题—机制—分析—实证—边界”排列。先说明一般配置问题，再说明 portfolio 实例。结果句等受控验证完成后填写，不能先承诺提高收益、降低风险或改善样本效率。

**所需证据：** C1–C3 的最终完成情况，以及预先确定的主要指标。现有最高样本均值只能作为探索性发现。

### 1. Introduction and Contributions（约 650 词）

**中心论点：** 结构化映射解决动作可行性，但没有自动解决依赖分支如何学习共同收益与风险的问题。

建议内容：

- 从有限资源配置、调整成本和累积风险引出问题。
- 解释分解动作中“分支独立表现好”未必等于“对最终可行配置贡献大”。
- 区分已有方法：CAOSD 提供动作构造，RCPO 提供风险压力调节思路，difference rewards／COMA 提供信用分配动机。
- 明确本论文研究的是可行映射和路径记忆共同存在时的信用定义及验证。
- 给出 C1–C3，并按证据状态写贡献，避免将基础方法再次命名为新算法。

**所需证据：** [L1]–[L6] 的最近邻工作定位；映射耦合和历史记忆的可检验例子；当前实现确实具有相应接口。

**衔接：** 读者理解为什么信用值得研究，下一章准确规定动作、信息、轨迹和控制目标。

### 2. System Model and Problem Formulation（约 850 词）

**中心论点：** 这是一个同时包含逐动作支持限制与期望平均路径成本限制的序贯配置问题；portfolio 只是其中一个具体动态与成本模型。

建议内容：

- **2.1 Allocation system**：N 维标准单纯形、两个可重叠组的最低份额、补集转换及可行集非空假设。当前神经 actor 要求各分支支持非空，放在方法的实现条件中。
- **2.2 Controlled evolution and payoff**：先建立一般环境与内生记忆接口，再用 portfolio 的目标权重差、净增长和对数奖励说明调整成本。
- **2.3 Path cost and information**：明确状态与观察；previous allocation 与 path statistics 进入完整状态；当前 actor 不接收潜在市场状态、绝对财富或财富比率。
- **2.4 Policy objective**：折扣奖励、期望平均成本、硬可行政策类；portfolio 以 c-alpha 定义平均残差目标。风险阈值与 cost allowance 分开，不把它们写成逐路径保证。

保留财富数值保护、自融资近似和当前／最大回撤的准确区别；常数与详细核算留在附录。本章不定义分支 PPO 损失或 CF 训练算法。

**所需证据：** 环境代码 [I1]、映射代码 [I2]、CMDP 文献 [L3–L4]、金融核算文献 [L7]。

**衔接：** 问题已经明确，下一章给出所借用的结构化策略与信用思想。

### 3. Preliminaries（约 650 词）

**中心论点：** CAOSD、Lagrangian learning 和 counterfactual credit 属于相关但不同的方法线索；当前研究需要连接它们，而不是宣称已经统一其理论保证。

建议内容：

- **3.1 Feasible simplex decomposition**：四个支持集、组合质量与可行性命题；充分归属 CAOSD，完整证明移至附录。
- **3.2 Adaptive cost learning**：CMDP／Lagrangian、RCPO 思路与 PPO surrogate 的最少背景；实际乘子频率与损失留在方法。
- **3.3 Credit assignment with dependent decisions**：共享全局信用、standalone 和 difference rewards 的区别；COMA 的条件期望基线与本项目固定中性替换的区别。

virtual-agent 只作直观解释，不发展未实现的 Markov game、独立智能体或分散执行理论。

**所需证据：** [L1–L6]。PASPO 用于定位更一般可行空间，不声称本文已实现它；COMA 用于定位信用思想，不用于证明本项目无偏性。

**衔接：** 将背景中的概念转成精确的学习接口，并把 CF 作为方法章节的重要组成部分。

### 4. Proposed Method（约 1,800 词）

**中心论点：** 在实际可行映射内部构造中性替换配置，并分别维护路径状态，能够定义不同于 standalone 的信用信号；它的解释与学习目标必须一起说明。

#### 4.1 Feasible autoregressive allocation policy

从 surrogate action 到环境动作的确定性映射切入，给出自回归因子分解、联合与条件似然、训练掩码和 actor 支持条件。硬可行性来自映射；具体网络族和优化器限定可学习的策略。

**所需证据：** [I2–I3]；已有 CAOSD 命题。这里不新增整体最优性或策略类等价结论。

#### 4.2 Counterfactual credit within the feasible map

把 CF 正式定义为可行组合间的比较：

\[
\boldsymbol a_t=\mathsf T(\boldsymbol y_{1,t},\ldots,\boldsymbol y_{4,t}),\qquad
\boldsymbol a_t^{-i}
=\mathsf T(\boldsymbol y_{1,t},\ldots,\boldsymbol y_i^0,\ldots,\boldsymbol y_{4,t}).
\]

说明为何要重新计算组合质量、为何选 uniform neutral、哪些变量保持实际已采样值，以及 reward／cost shadow 如何各自递推。对照 standalone 的独立投资路径，形成一个明确的信用定义表。uniform 是当前实现的基准选择，不是理论最优替代动作。

**所需证据：** [I1–I3]；与 [L5–L6] 的精确区别；第 5 节的基准敏感性与机制验证。

#### 4.3 Mapping coupling, trajectory interpretation and analytical properties

本节是建议新增的**分析任务**，不是已完成的证明或发现。重点是小而准确的结构分析，不追求未经支持的整体收敛定理：

| 分析项 | 应建立的内容 | 贡献归属 |
| --- | --- | --- |
| 替换后可行性 | 中性分支仍在其支持单纯形内，经同一映射后配置满足约束 | CAOSD 可行性结论的推论，不能当作新理论主贡献 |
| 映射耦合 | 改变 branch 2 可同时改变后续组合质量，不能一概用自身质量乘分支差代替完整配置差 | 与当前构造直接相关的局部分析，需整理推导及例子 |
| 路径状态分解 | 持久 shadow 的差异同时包含当前配置变化和历史内生状态差异 | 用来界定信用含义，不将代数恒等式包装成重大新定理 |
| 基线抵消条件 | 在自回归依赖下，后续实际动作可携带被替换分支的信息；一般无偏基线性质不能直接移植 | 给出明确条件或简洁反例，划清理论边界 |

可考虑整理的映射关系：在其他分支值固定、两组 CAOSD 的精确算术条件下，

\[
\Delta\boldsymbol a_2
=z_2(\boldsymbol y_2-\boldsymbol y_2^0)
 (z_3-z_3^{-2})(\boldsymbol y_3-\boldsymbol y_4).
\]

第二项展示了替换通过组合质量产生的耦合。这是待正式呈现的结构关系，不是已验证的性能原因。

路径解释则从下式出发：若 f 表示相同外生输入下的 reward 或 cost 计算，h 与 h^{-i} 是 actual／shadow 的内生记忆，则

\[
f(h,a)-f(h^{-i},a^{-i})
=[f(h,a)-f(h,a^{-i})]
+[f(h,a^{-i})-f(h^{-i},a^{-i})].
\]

该式区分同状态动作比较与历史状态差异，不赋予两项无偏因果解释。还应说明：当前配置差为零，并不必然令持久路径的成本差为零。

**所需证据：** 映射代数、最小可核验例子及 [I1–I2]；不能用总收益排名替代理论论证。

#### 4.4 Risk-penalised learning and branch objectives

统一列出 global、standalone、CF-R、CF-C、CF-RC 的实际信号、归一化顺序、外部质量权重、掩码和熵正则。说明 critic 得到的状态信息、signed branch cost 与实际非负 cost 的不同角色。

全局 multiplier 只由 actual rollout 平均残差更新，PPO epochs 内固定。说明折扣 cost advantage 与平均成本约束的近似关系。现有 CAOSD、PPO、RCPO 均保留归属。

**所需证据：** [I3–I4]；共同 surrogate 下的消融及原始模式比较。只写当前实现，不补入 expectation baseline、closed-loop resampling 或新风险准则。

#### 4.5 Algorithm and computational requirements

用一份紧凑伪代码串起采样、映射、actual／benchmark／shadow 更新、优势估计、固定 multiplier 的策略更新和一次 rollout 后的 multiplier 更新。

明确 CF 需要可重算的奖励／成本与内生状态转移，或足够的模拟访问。portfolio 的外生市场收益能共享；一般环境若状态受动作影响，shadow 必须更新自己的状态，不能直接复制实际状态。报告额外映射次数、记账、critic 规模、运行时间与内存；具体开销待测，不承诺免费或线性加速。

**衔接：** 方法的各项选择产生可检验问题，实验围绕这些问题组织，而非只按版本罗列表格。

### 5. Experimental Evaluation（约 1,700 词）

**中心论点：** 判断 CF 的价值需要同时比较效用、成本残差、动作可行性、稳定性及计算代价，并区分方案整体差异与信用信号自身效应。

建议子节：

- **5.1 Protocol, metrics and reproducibility**：任务、数据划分、预算、种子、checkpoint 规则和统计单位。
- **5.2 Controlled credit comparison**：共同学习条件下的 global／standalone／CF 比较；原 V2.6/V3.2 作为归档训练方案结果单独呈现。
- **5.3 Mechanism and sensitivity analysis**：映射耦合、组重叠、shadow 历史、中性替换、损失权重与熵正则。
- **5.4 Robustness and external validation**：actor 分布、风险预算、市场状态及未见任务／历史数据。
- **5.5 Computational trade-offs**：收益与成本变化是否值得额外模拟、网络和优化开销。

**所需证据：** 第 5 节实验矩阵中的验证；现有 [D1] 只覆盖部分探索性问题，不能替代这些受控证据。

**衔接：** 讨论从结果解释适用条件、成本和失败模式，不重复列举最佳数值。

### 6. Discussion and Limitations（约 600 词）

**中心论点：** CF 的价值是有条件的；应解释哪些几何与路径条件改变了信用，而不是把 mixed results 归纳为普遍提升。

讨论四个方面：映射耦合与 standalone 的目标偏离；影子历史的作用和解释成本；风险收益换手及计算代价的联合权衡；从已验证环境向其他配置系统外推的边界。

若一般性只来自形式模型，明确写成形式适用范围。若新增非金融任务成功，才讨论相应跨任务证据。单种子、合成市场、目标权重成本代理和 deterministic decoding 的限制按最终证据更新。

**所需证据：** 第 4 节分析与第 5 节实际结果。无需逐段重复免责声明，也不从不显著的提升推出算法优越。

### 7. Conclusion（约 200 词）

**中心论点：** 直接回答研究问题，分开总结已证明的性质、已观察的效果及尚未解决的部分。

结论不再以“某一版本均值最高”作为唯一落点。若 CF 仅在特定风险或几何条件下有益，就明确给出条件；若没有稳定收益，则报告其机制诊断价值，并重新评估是否仍具备足够投稿增量。

**所需证据：** C1–C3 的最终覆盖情况，不能引入新结果。

### Appendices

放置 CAOSD 证明及归属、局部分析细节、完整损失／超参数、金融执行与数值保护、数据和种子协议、完整结果。具有决定性的负面结果保留在主文；不能用附录隐藏不利证据。

## 5. 待确认的实验与证据矩阵

**本节全部是计划，不是执行指令；本轮未运行任何项目。**

| 编号 | 要回答的问题 | 最小设计 | 对应主张与当前状态 |
| --- | --- | --- | --- |
| E0 | 归档指标是否符合当前定义？ | 从可用 checkpoint 重评或完整收益序列重算含初始财富的 MDD；核验 benchmark、费用、c-alpha 和 checkpoint 身份 | 投稿前先完成；代码已修复不等于归档结果已更新 |
| E1 | 信用信号是否在可比较条件下有价值？ | 同一 actor／映射／对偶／数据协议；共同 conditional branch surrogate 下比较共享全局信用、standalone、CF-R、CF-C、CF-RC | 主要作用归因；当前历史结果不能完成此判断 |
| E2 | 差异是否来自权重、熵或 critic？ | 固定主要比较的 branch weighting、熵规则、归一化与信息协议；另做质量权重及熵的交叉消融，并记录 critic 规模与输入 | 必须把现有 z／mask 差异拆开，不能只令 entropy coefficient 相同 |
| E3 | 训练随机性是否改变结论？ | 多个独立训练种子；在相同 test scenarios 上配对；不报告 best seed | 现有一个训练种子不足；可从 5 个种子做预算规划，最终按方差与算力决定，不把 5 当作充分性定理 |
| E4 | 结构解释是否对应实际机制？ | 选少量预先规定的组重叠／阈值和状态历史设置；比较完整重新映射与仅解释自身质量时的偏差；记录配置差及 reward／cost 差 | 支撑第 4.3 节；配置差为零与路径信用为零分开统计 |
| E5 | 结果是否只依赖一个合成生成器？ | 增加独立外部验证；portfolio 优先采用有许可、可复现的历史数据和 chronological walk-forward | 财经应用证据；具体数据集、授权与调参预算待确认 |
| E6 | 一般配置主线是否有跨任务证据？ | 推荐增加一个非金融配置任务，如有调整成本与累积 backlog cost 的工作负载分配；明确 demand 外生、queue 状态由各自 shadow 更新 | 跨领域主张需要此类证据；该环境尚未实现，不能在当前稿中声称已支持 |
| E7 | CF 的收益是否抵偿额外开销？ | 同时报告环境交互量、shadow 计算、优化次数、墙钟时间和内存；给出性能—成本比较 | 避免从等交互预算推出等计算预算或样本效率优势 |

### E1／E2 的关键设计约束

1. **区分两种比较。** V2.6/V3.2 保留为 training-scheme comparison；另设计 controlled credit comparison。新受控 baseline 不能仍用原 V2.6 名称却悄悄改变目标。
2. **区分联合与条件 surrogate。** joint-likelihood PPO/RCPO 是有价值的整体参照，但其与 conditional branch PPO 的差异不能归为 CF 效应。主要信用消融在共同分支 surrogate 下进行。
3. **先定义质量尺度，再训练。** standalone 信号与 portfolio difference 的尺度不同；若某种质量权重是信号定义的一部分，应在方法中明确，不能在结果出现后调整来制造公平性。
4. **不要过度控制掉机制。** 对 critic 输入给出明确、可复现的信息协议；记录容量和计算。哪些输入差异属于信用机制本身，哪些属于额外模型能力，要分别讨论。
5. **匹配算法比较。** 若要归因于 adaptive multiplier，需在同一 actor、credit、epochs 和 learning-rate schedule 下比较固定／自适应风险压力。当前 Gaussian PPO 与 RCPO 的既有训练设置不满足这一纯消融条件。

### 统计与评估口径

- 主要风险指标包含平均 c-alpha、其不确定性以及路径可行率；MDD 和回撤超限成本不互相替代。
- 区分独立训练种子与市场延续。不能把 seed × continuation 所有行当作独立样本来人为缩窄区间；采用种子层面汇总或与采样层次匹配的分层区间估计。
- 对共享市场进行配对比较，报告效应量和区间；事先确定主要比较，区分确认性比较与多指标探索。
- 固定 validation checkpoint 规则，不在 test 上挑 actor、种子、风险预算或最好变体。若报告不同选择规则，完整披露。
- 继续用 deterministic decoding 复现历史比较；若要讨论 stochastic policy 的约束值，必须额外评估相应随机策略。
- 风险目标未满足时报告 violation magnitude，不将“风险较低”改写为“满足约束”。
- 可复现协议记录代码版本、配置、初始状态、训练预算、市场／数据划分、checkpoint 和结果生成程序。

### 工作量控制

先完成 E0 与分析，再用有限的 validation-only pilot 确定新的统一训练预算并锁定协议；随后进行受控训练及外部检验。不机械复制原来的 60,000-update 预算到所有消融，不根据测试表现缩放预算。先固定主 actor 配置并说明依据，另一分布用于稳健性检验；选择不依据本轮测试结果。

## 6. 当前证据能支持什么

本轮读取的 V2.6/V3.2 汇总仍是一个训练种子、20 条共享合成市场延续下的归档结果：

| 已观察事实 | 现在可写的解释 | 不足以支持 |
| --- | --- | --- |
| Dirichlet CF-R 相对财富约 8.18%，standalone 约 8.06%；已有配对区间覆盖零；换手约 24.00% 对 14.27% | 较高样本均值伴随交易行为变化，值得受控追查 | 显著收益改善、稳定优于 standalone |
| Gaussian CF-C 的样本平均成本残差约 0.00650，standalone 约 0.00958；换手约 7.22% 对 1.91% | 存在风险成本与换手的条件性权衡线索 | 已满足平均风险预算、已隔离 CF 成因 |
| 全部八个 return-best RCPO 策略的 pooled empirical c-alpha 为正 | 当前 deterministic evaluation 未达到其样本平均残差目标 | 对 stochastic training policy 或总体分布作已证实的约束结论 |
| CF-RC 没有一致优势 | 收益信用与成本信用不能假定具有叠加收益 | 两种差分信用联合必然最好 |
| MDD 计算代码已包含初始财富，归档统计尚未重评 | MDD 相关排名与区间继续作为待核验结果处理 | 将旧 MDD 改善当作投稿主结论 |

这些事实用于提出问题，不用于在新大纲中预写正面答案。

## 7. 章节—证据映射

| 章节 | 服务的子问题 | 主要文献／实现／实验 | 当前缺口 |
| --- | --- | --- | --- |
| Introduction | RQ1–RQ4 | L1–L6；I1–I4；D1 | 定向相邻文献检索与增量定位 |
| Problem Formulation | RQ1、RQ2、RQ4 | L1–L4、L7；I1–I2 | 根据最终任务确认成本与状态假设 |
| Preliminaries | RQ1–RQ3 | L1–L6 | 维持紧凑、明确归属，不虚构理论继承 |
| Method 4.1–4.2 | RQ1、RQ2 | I1–I3；L1、L5–L6 | 完整定义与可复现接口 |
| Method 4.3 | RQ1、RQ2 | 映射推导、路径分解、反例／条件 | 正式分析尚待整理 |
| Method 4.4–4.5 | RQ3、RQ4 | I3–I4；L3–L4 | 受控目标设计、开销测量 |
| Experiments | RQ1–RQ4 | E0–E7；D1 | 新的受控、跨种子与外部证据 |
| Discussion／Conclusion | RQ1–RQ4 | 已完成分析与最终实验 | 按实际结果选择结论强度 |

### 实现与数据定位

- **I1：环境、信息与 shadow 记账。** [env.py](<C:/Users/miaoj/Desktop/RCPO in trading/src/rcpo_portfolio/env.py:299>)；反事实路径更新见 [相应实现](<C:/Users/miaoj/Desktop/RCPO in trading/src/rcpo_portfolio/env.py:794>)。
- **I2：两组映射及质量耦合。** [simplex.py](<C:/Users/miaoj/Desktop/RCPO in trading/src/rcpo_portfolio/simplex.py:111>)。
- **I3：分支损失、熵正则和 critic。** [ppo.py](<C:/Users/miaoj/Desktop/RCPO in trading/src/rcpo_portfolio/algorithms/ppo.py:237>)；[models.py](<C:/Users/miaoj/Desktop/RCPO in trading/src/rcpo_portfolio/models.py:179>)。
- **I4：奖励／成本优势与实际 multiplier。** [rcpo.py](<C:/Users/miaoj/Desktop/RCPO in trading/src/rcpo_portfolio/algorithms/rcpo.py:63>)。
- **D1：当前证据。** [归档汇总](<C:/Users/miaoj/Desktop/RCPO in trading/evaluation/section9_simplex_v3.2_policy_comparison/section9_comparison_summary.json>)、[逐 continuation 指标](<C:/Users/miaoj/Desktop/RCPO in trading/evaluation/section9_simplex_v3.2_policy_comparison/section9_branch_metrics.csv>)和[当前论文](<C:/Users/miaoj/Desktop/RCPO in trading/reports/final_project_report.tex>)。这些材料不等于完整训练随机性证据。

### 核心文献及使用范围

- **L1：Winkel et al., ECAI 2023, Simplex Decomposition for Portfolio Allocation Constraints in Reinforcement Learning。** 用于两组几何、映射和策略背景。[原文](https://arxiv.org/html/2404.10683v1)
- **L2：Winkel et al., NeurIPS 2024, Autoregressive Policy Optimization for Constrained Allocation Tasks。** 用于一般 allocation 与更广可行空间的最近邻定位；不宣称已实现 PASPO。[会议页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d79c1390baa2e4835586b094d82e5ffb-Abstract-Conference.html)
- **L3：Achiam et al., ICML 2017, Constrained Policy Optimization。** 用于 CMDP 奖励与成本约束框架。[原文入口](https://proceedings.mlr.press/v70/achiam17a.html)
- **L4：Tessler et al., ICLR 2019, Reward Constrained Policy Optimization。** 用于 penalty／multiplier 思路和原方法假设。[原文](https://arxiv.org/html/1805.11074v3)
- **L5：Foerster et al., AAAI 2018, Counterfactual Multi-Agent Policy Gradients。** 用于条件期望基线和信用思想的最近邻区别。[原文](https://arxiv.org/html/1705.08926v1)
- **L6：Castellini et al., Difference Rewards Policy Gradients。** 用于直接差分奖励与策略梯度的已有先例，避免把默认替换思想写成新发明。[论文与出版信息](https://arxiv.org/abs/2012.11258)
- **L7：Jiang, Xu and Liang, 2017, A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem。** 用于财富、配置和费用核算的参照。[原文](https://arxiv.org/html/1706.10059v2)
- **补充邻近工作：Nguyen, Kumar and Lau, NeurIPS 2018, Credit Assignment For Collective Multiagent RL With Global Rewards。** 用于资源与集体决策中的信用定位；不等同于本项目共享参数的自回归分支。[会议页面](https://papers.neurips.cc/paper_files/paper/2018/hash/94bb077f18daa6620efa5cf6e6f178d2-Abstract.html)

文献与 scope 入口于 2026-11-02 核对；这不是系统综述或完整新颖性／撤稿审计。Ding et al. 的“系统模型—问题重构—学习方法”顺序仍可作为叙事参照，不作为本项目理论保证的依据。[参考论文](https://arxiv.org/pdf/2104.06007)

## 8. 保留、合并、删除与附录安排

| 处理 | 内容 |
| --- | --- |
| 保留 | 一般 N 维配置主线、两类约束分工、真实观察／状态区别、portfolio 记账与风险、CAOSD 归属、实际 RCPO 更新 |
| 提升为主线 | CF 的问题动机、可行映射内定义、路径解释、机制分析与受控验证 |
| 合并 | standalone 与 virtual-agent 的长篇解释，作为信用参照；重复的风险与因果边界集中到方法解释和讨论 |
| 重构实验叙事 | 从“V2.6 对 V3.2 排名”转为“研究问题—控制设计—效应与开销”，保留完整版本溯源 |
| 删除 | 以模块组合本身作为新算法的语言、无法支撑的普遍优越性、没有相邻文献支撑的首次声明、未实现 MARL 理论 |
| 移至附录 | 已有可行性证明、长推导、实例常数、数值保护、完整配置和扩展结果；主要不利结果仍在主文 |

建议主图表：一张映射与 actual／shadow 信息流图，一张配对效用—成本结果图，一张映射／历史机制诊断图，一张跨场景与计算开销图；三张表分别汇总主比较、关键消融及复现／计算预算。按证据需要取舍，本轮不生成图表。

## 9. 分阶段推进与结论分支

| 阶段 | 主要任务 | 可以获得的结论层级 |
| --- | --- | --- |
| A：整理与核验 | E0、最近邻定位、公式及分析任务、复现记录 | 机制定义和解释正确；不能推出性能优越 |
| B：核心研究 | E1–E4、多种子及预先固定协议 | 判断 CF 信号／方案是否在指定条件下产生稳定价值 |
| C：投稿强化 | 历史 walk-forward、计算开销；跨领域主张对应非金融任务 | 扩大经验证据的适用范围，仍按实际任务限定结论 |

结果应允许三条结论分支：

- **稳定改善：** 仅对受控比较、实际任务与指标声明收益，并报告代价。
- **条件有效：** 将贡献落在可验证的适用条件和机制解释，不写普遍优越。
- **没有稳定收益：** 不能通过标题和措辞维持“有效算法”叙事；评估负面／边界分析是否足够独立、有用，否则收缩 CF 的贡献强度。

一般性主张与非金融验证的关系：若不实现 E6，可保留一般数学模型，但经验结论明确限于 portfolio；不能声称已证实跨领域有效。优先目标应是得到可信、可复现的研究结论，再决定发表定位。

## 10. 篇幅预算与确认点

| 章节 | 建议英文词数 |
| --- | ---: |
| Introduction | 650 |
| Problem Formulation | 850 |
| Preliminaries | 650 |
| Proposed Method | 1,800 |
| Experimental Evaluation | 1,700 |
| Discussion | 600 |
| Conclusion | 200 |
| 正文合计 | 6,450 |

摘要约 200 词，参考文献与附录另计；最终服从作者确认的 venue 要求，不以拉长篇幅替代证据。

待作者确认的是：是否采用“CF 定义与机制分析 + 受控证据”的中心叙事，以及是否进入后续分析／实验规划。具体 venue、算力预算、数据集和新增任务尚未确认。

**本轮在此停止。该大纲不授权正文修订、代码变更、新实验或投稿动作。**
