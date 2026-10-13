# Waterbirds 论文验证方案（待确认）

执行更新（2026-10-05）：最新五 Agent＋本地构建成对 OOD 已完成，保留结果为 `runs/waterbirds-shortcutminer-paired-20261005-v1`，见 `docs/waterbirds-shortcutminer-core-run.md`。此前试跑和独立模型结果已按用户要求清理。下文仅为历史方案；当前配置为 `configs/experiments/waterbirds-shortcutminer-background-formal.json`。promotion gate 校准、受控比较和完整论文 E1–E5 不能因此全部标为通过。

状态：proposal，不是 frozen，不是已经完成的论文实验。2026-10-04。
本轮仅形成方案，不新增真实 Provider 调用，不改科学规则或已有结果。

## 目标与结论边界

本方案把 Waterbirds 从一次成功结束的 metadata audit，推进为可复现的
图像模型验证个案。完整证据可以支持、反驳或保持不确定；不会以“确认
shortcut”作为必须达到的程序成功条件。一个 Waterbirds 个案不代表整篇
论文的 E1–E5、基线和消融均已完成。

此前开发运行只执行 metadata audit，未完成 Stage-2；现已按用户要求删除。
它不是正式结果，也不用于调整待冻结阈值。

## 已核实的数据条件

根目录：`E:/ShortcutMiner-Experiments/waterbirds`。
元数据：`metadata.csv`。

| 项目 | 核实结果 / 拟冻结角色 |
|---|---|
| 总行数 | 11,788 |
| 图像缺失 / 越出数据根目录 | 0 / 0 |
| 重复 img_id / img_filename | 0 / 0 |
| 样本标识 | img_id，控制字段，不作科学候选 |
| 图像路径 | img_filename，控制字段，不作科学候选 |
| 标签 | y |
| 划分 | split：0=train，1=validation，2=test |
| 划分规模 | 4,795 / 1,199 / 5,794 |
| metadata.csv SHA-256 | 2f023b9dc371f6fbb82c7cb810ff0f03e19c54c8055a2e37517ae7bb56644362 |

角色与 split 编码来自原始实现，并与本地文件核对：[原始数据加载器](https://raw.githubusercontent.com/kohpangwei/group_DRO/master/data/cub_dataset.py)。
上表只核实存在性与身份，不代表已逐张完成解码或全量图像哈希。
正式开始前还要建立全量图像内容 manifest、验证解码和固定数据视图。

## 推荐的实验协议

### 1. 数据暴露与划分

默认方案是明确标注的 **metadata-assisted audit**，不是 blind discovery。
发现、Stage-1 和 promotion 只使用 train 数据；validation 用于预注册的
模型选择；test 留到最终评估，不能参与候选发现、阈值调整或选 checkpoint。
不更改官方划分，不重新随机混合 11,788 行。

`place` 可作为该模式的公开训练集元数据；不能把它描述成系统独立发现
且从未看到的答案。`place_filename` 不作为可学习特征或科学候选，避免
来源路径泄漏。评估端的 (y, place) 分组定义不写入 Identify taxonomy 或
候选示例。若论文主张 blind discovery，需要另行冻结隐藏分组元数据的
发现视图；当前已暴露完整 metadata 的旧运行不能用作盲测。

原始数据保持不变，派生视图及其来源映射必须保存并哈希。

### 2. Stage-1 与证据提升（建议值，尚未校准）

保持 max_candidates=5，不改变发现数量或科学校验。
每个可绑定的元数据候选预先指定一个主要关联检验。

建议的资源路由 gate：

- statistical_check 的 p_value <= 0.01；
- valid_count >= 200；
- metric_rules 仅这一条，组合仍为现有 any_rule_satisfied；
- 缺失、失败、无适用数值证据：deferred_insufficient_evidence；
- qualitative_promotion_rule 仍为 unavailable。

0.01 是针对最多 5 个预先指定主要检验的候选 Bonferroni 预算 0.05/5，
不是 ShortcutMiner 已校准的 cutoff。必须检查有效 p-value 的前提与
检验实现；若每个候选有多个主要检验，不能继续沿用这个预算。
minimum_samples=200 是拟议的路由可靠性/资源要求，不是现有 fixture
阈值或普适统计保证。正式冻结前用独立开发数据和负控检验 gate 的
误提升、漏提升与运行成本；不能用当前 Waterbirds test 反调数值。

不统一对 effect_size 设数值门槛：当前后端可能返回 Cramer's V、eta²
或有符号 Pearson r，它们不是同一种量。

这个 gate 只决定是否花费 Stage-2 资源，不确认 shortcut。纯视觉候选若
没有适用的 Stage-1 数值证据，在现有机制下仍会 defer；不能通过伪造
元数据绑定、LLM confidence 或放宽校验让它们通过。若论文要求这种
候选也进入模型验证，需另行设计、验证并批准 gate 扩展，不能靠配置
宣布当前实现已经覆盖。

### 3. Stage-2：真实任务模型

推荐主方案采用两个 ImageNet 预训练模型：ResNet-18 和 ResNet-50，均做
CUDA ERM 全模型微调。每个模型 3 个种子，共 6 次独立训练；两个模型
使用相同种子、数据划分和评估协议，分别汇总，不混合成一个均值。
ResNet-18 提供较小模型对照，ResNet-50 保留与原始 Waterbirds 研究
可比较的
模型家族基础：[原始实现](https://github.com/kohpangwei/group_DRO)。

下列是本项目拟议配置，不声称逐项复现原论文：

| 参数 | 建议冻结值 |
|---|---|
| 模型 | torchvision ResNet-18、ResNet-50，各自 ImageNet1K V1 初始化 |
| 输入 | 224×224 RGB，ImageNet normalization |
| train 变换 | RandomResizedCrop(scale=0.7–1.0)，水平翻转 |
| val/test 变换 | Resize 256，CenterCrop 224 |
| 损失 / 优化器 | CrossEntropy / SGD，momentum=0.9 |
| 学习率 / weight decay | 0.001 / 0.0001 |
| batch size | 32，先做显存 capacity acceptance |
| 最大 epochs | 100；按 validation 总体 accuracy 选 checkpoint，固定 tie-break |
| 随机种子 | 每个模型 42、43、44；2×3=6 次独立训练 |
| 验证/测试预测 | 同一 checkpoint，按 img_id 严格关联 |
| 初始化身份 | torchvision exact version + 每个模型的权重 SHA-256 |

训练不使用 place 或 place_filename 作为模型输入，不按 test 指标选择
checkpoint。上述训练配置先在开发阶段检验收敛，再在正式运行前冻结；
冻结后改变训练 recipe 要形成新协议版本，不能挑选最好的 test 运行。

当前 `ReferenceImageLinearClassifier` 只是 16×16 RGB logistic regression
参考后端；仓库文档明确它不是 paper baseline。不能改一个名称就把它
变成 ResNet。实施主方案需要新增 allowlisted 任务模型执行适配器、
安全 checkpoint 格式和同模型 provenance；五个 Agent 与 LangGraph
仍可保留。这个新增后端超出上一轮纯环境冻结范围，需要你确认。

可选的效率方案是固定 ResNet-50 特征后只训练线性 head，但必须以
linear-probe 协议命名，不能声称等价于全模型 ERM 或原论文结果。

### 4. 模型验证与受控干预

每个训练 seed 至少保存总体、balanced、四组 accuracy、worst-group
accuracy，以及逐样本预测、group counts 和 95% CI。另存按训练组比例
加权的 adjusted average accuracy，避免把不同平均定义混在一起。
四组与加权指标参照 [WILDS 的 Waterbirds 评估实现](https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/waterbirds_dataset.py)。

每个模型的 3 个 seed 分别报告，再分别汇总 mean/std；训练随机性与 prediction bootstrap
不确定性分开。bootstrap 建议 1,000 次（现有参数上限内），分组分层抽样；
干预比较在同一样本上成对抽样。任何新增统计方法先离线验证。

关联与 group gap 不能单独证明背景依赖。若要主张模型依赖 suspected cue，
需要对同一 checkpoint 的配对、标签保持干预，并保存变换规格、mask、
原/变换预测、样本身份、CI、失败/排除清单与 provenance。

当前目录只有合成图片和 metadata；背景来源字符串不是前景分割 mask。
不能用任意裁剪或全图颜色变换冒充“只改背景且鸟不变”。正式背景干预
需要经检查的前景 masks/原始生成资产；没有这些资产时如实报告模型
行为/组差异，不能声称已完成背景因果验证。OOD 也不能用修改 test 标签
或重新打乱分组替代真实输入条件改变。

Qwen 仅承担有样本来源的定性观察，建议每个路由任务最多 8 张样本，
含原始/干预配对。观察数量不足、标签保持未验证或图像读取失败均保存
为限制；Qwen 的语言结论不作为自动提升或确认阈值。

### 5. 可复现性与完成判据

Windows conda shortcutminer 保持现有 exact lock。训练需要独立验收 GPU、
batch、数据加载和恢复；不随意替换环境依赖。/runs 的原约定落实为
`E:/ShortcutMiner-github/runs`，API keys 仅由 .env.formal 运行时加载。

正式数据视图、policy、模型配置、初始权重、全量数据内容均哈希。
按你的既定要求，正式 E1–E5 必须先由你手工完成 clean Git commit，
candidate manifest 与 Windows capacity acceptance 也要完成；不自动 commit。

完成验收需要：

1. 显式 img_id/img_filename/split 接线，跨阶段一致，控制字段排除。
2. 离线端到端测试覆盖图像/Qwen fake transport、promotion、Stage-2、resume。
3. 真实容量验收与小规模开发运行通过后冻结配置。
4. 正式重复运行：每次独立 run_id，归档全部调用/工具/模型/测量记录。
5. 每个候选列出实际可执行检查、完成或阻塞原因，不能要求全部 promoted。
6. 论文表分别报告发现、association、model behavior、intervention 和成本；
   未有证据的单元保留 NA/未验证，不能由 Summarize 补全。
7. Waterbirds 个案通过后，整篇论文的其他数据集、基线、消融、E1–E5
   仍按既有论文计划执行，不在本方案中臆造或替换定义。

## 本次需要确认

建议批准：metadata-assisted 模式；上述待校准 Stage-1 gate；新增 CUDA
ResNet-18/ResNet-50 ERM 任务模型后端；每模型 3 个 seed、共 6 次训练；
8 张/定性任务预算。代码范围仅
数据接线、显式运行配置、allowlisted 模型执行后端及其测试/文档，不
改变五个 Agent 的职责、LangGraph、已有科学验证规则或结论阈值。

另外请明确：是否要求背景因果干预作为本个案必备完成条件；若要求，
需要提供可验证的 masks/生成资产或确认另外准备这些资产。没有资产
不能保证该部分可以完成。
