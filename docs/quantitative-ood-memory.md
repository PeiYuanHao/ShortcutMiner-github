# Quantitative Agent 的 OOD 配方记忆

论文第 III-C 节的自主性边界是：人提供可复用的干预模块和约束，Quantitative Agent 根据候选及已有证据选择配方，执行层验证并运行确定性脚本。不能把实验配置直接选定的干预称为 Agent 自主选择。

## 当前接入

配方库为 `agents/quantitative/workspace/ood_recipes.json`，策略说明为同目录的 `MEMORY.md`，版本为 `ood-memory-v1`。本轮保留五 Agent、LangGraph 图、科学阈值、Stage-1/Stage-2 分工和 EvidenceLedger/SummaryReport 权威，仅在 Quantitative 执行边界增加一个受约束的规划子调用；没有新增第六个 Agent。

新配置 `configs/experiments/colored-mnist-irm-shortcutminer-memory-v2.json` 不含预选的 color_ood_specs。`ood_recipe_selection.fields` 只授权候选可验证的字段，不指定操作。现有模型、种子、数据划分、δ=2 个百分点及 5% 误判预算保持不变。

执行顺序：

1. Identify 发现候选，Executive 路由，Stage-1 收集便宜证据，原有 promotion gate 控制是否进入 Stage-2。
2. 执行层只检查训练分区的图像，生成像素格式和配方适用条件。它不读取验证、ID/OOD 图像，不计算预期 OOD 效果。
3. 在任何昂贵训练前，Quantitative 接收当前候选、当前任务的成功便宜证据及符合条件的记忆配方，返回配方 ID、授权字段、选择理由、所引用 ToolCall ID。
4. 执行层拒绝不支持的配方、未授权字段、其他任务的证据及任意原始参数，随后将选择编译为严格类型的训练／干预计划。普通实验 ToolCall 仍只接受 plan_step_id。
5. 原有试探早停决定是否扩展六模型；配对与独立 OOD 使用同一组验证集选定的权重，最终统计排除试探组。

不存在默认颜色交换回退。无适用配方、无合适选择或选择校验失败时，停止本项执行，不能虚构 OOD 结果。Agent 只选择一个主要配方，不能比较多个 OOD 结果后挑选最大降幅。

## 配方覆盖与实际能力

| 方向 | 配方 | 自动选择状态 |
|---|---|---|
| 红绿前景颜色 | 交换红绿通道 | 已接入；仅限经过像素校验的黑底红绿前景 |
| 红绿前景颜色 | 去色：将 max(RGB) 数字强度复制到三通道 | 已接入；保留数字形状及强度，与加权灰度转换不同 |
| 饱和背景颜色 | 循环颜色通道 | 已接入；要求一个饱和通道、另外两个相同的灰度内容通道 |
| 自然背景 | DINO/SAM/ViTMatte 掩码后中性化 | 已接入工具池：Agent 选择已授权的本地模型链及前景提示，训练预览与逐图质量检查保留失败 |
| 自然图像前景颜色 | 掩码内 LAB/Reinhard 颜色迁移 | 尚未实现；不能用全图通道交换冒充 |
| 常量角落补丁 | 去除／位置随机化 | 已接入；训练图像自动绑定唯一常量角块的尺寸与颜色，Agent 选择操作，严格保留四个角块以外的像素 |
| 纹理 | 受约束纹理／模糊扰动 | 自动绑定尚未接入 |
| 编码、分辨率 | 配对重编码／重采样控制 | 自动质量约束绑定尚未接入 |

配方库可以预存所有这些方向，但只有具有执行后端、资源绑定和适用条件校验的条目才能执行。“写入 memory”本身不等于实现论文中的完整构建能力。

常量角落补丁的原始覆盖区域下面没有干净像素可恢复；去除只能中性填充，位置随机化也可能覆盖目标角落的自然内容。此限制随实验披露，不能将它称为无混杂的因果识别。位置随机化使用固定种子和角块外内容哈希，不访问标签；允许目标角落与原位置相同。预处理采用 spatial_preserving，不随机裁剪或翻转待检的位置线索。

## 审计和恢复

每个任务的 `quantitative/<task_id>/ood_recipe_selection.json` 保存选择输入、训练像素身份、catalog SHA-256、选择理由、证据引用、操作及记录哈希。恢复时复用原选择；输入、目录版本或记录变化则拒绝恢复，不重新选择以迎合结果。Provider 的调用另外由原有 model_calls telemetry 记录。

## 运行及历史结果

Windows conda `shortcutminer` 中下一次新审计：

```powershell
python scripts/run_colorized_mnist_shortcutminer.py --config configs/experiments/colored-mnist-irm-shortcutminer-memory-v2.json --live
```

运行器默认配置也已切换为这个版本。该命令使用 `.env.formal` 的既有运行时加载方式；没有 Docker 或 Linux 依赖。

此前 `colored-mnist-irm-shortcutminer-20261005-v1` 的六模型结果、原配置和源码归档保持原样，它采用配置预选红绿交换，不能追溯改称为 Agent 选择配方。修改后需在新的 run_id 下重新运行，才能得到包含真实选择记录的新实验结果。该历史改动当时仅完成离线测试；后续真实审计采用独立 run_id，见追加的构建质量说明。


## Construction quality and Agent-owned repair

Quantitative selection now includes training-only construction diagnostics and up to two quality revisions. Every attempt, choice, parameter binding, quality failure and final decision is saved in ood_recipe_selection.json. There is no deterministic substitute when an Agent chooses an invalid recipe. Revisions cannot use model predictions or held-out outcomes.

Legacy source-only corner filling fails the strict residual-position gate. Executable repairs neutralize all four corners and compare a matched original-position anchor against either patch removal or hash-assigned position. The seed/content hash does not use labels; target can equal source and realized proportions need not be exactly balanced. Both sides preserve every outside pixel; overwritten original corner content cannot be restored. Raw-to-anchor changes are separately reported. Sampling remains one intervention per independent protected-content group, so the original exact-binomial sequential interval and alpha/delta/budget are unchanged.

The available_local_foreground_tools pool authorizes hashed local DINO/SAM/ViTMatte tool chains without preselecting an OOD transformation. The Agent chooses a tool and foreground object prompt for natural backgrounds. Training-only mask previews retain failures; held-out masks must meet existing coverage and protected-region checks. New configs/experiments/waterbirds-shortcutminer-memory-v2.json demonstrates this binding; the old Waterbirds result is unchanged and has not been rerun.

The new corner config is configs/experiments/synthetic-corner-patch-shortcutminer-memory-v2.json. It optionally evaluates the existing pilot on independent test_ood after a deferred probe, explicitly as diagnostics with no expansion or final-confirmation claim. A processing baseline at or below validation majority accuracy causes a deferred processing decision rather than negative shortcut evidence.


## 向下兼容约定（2026-10-06）

| 配置 | 选择流程 | 配方库 | 配对统计 |
|---|---|---|---|
| v1：仅 fields / ood-memory-v1 | 冻结 legacy-v1 | 原字节恢复的 ood_recipes.json | 原精确区间、alpha/delta/checkpoints |
| 已运行 v2：显式 revisions / tool pool | 保留 quality-v2 | 原字节保留的 ood_recipes.quality-v2-compat.json | 原精确区间、原预算 |
| 新版显式 quality-v2 / ood-memory-v2 | 质量检查及 Agent 修正 | ood_recipes.v2.json | 原精确区间、原预算 |

没有版本开关的旧配置不自动迁移。已有 corner patch v1/v2 配置及所有历史 runs 文件不改写，已存选择记录可在对应版本直接校验复用，不重新请求 Provider。旧 prompt、响应 schema、配方库和选择执行逻辑单独保存；历史完整源码身份仍以各 run 的 source_snapshot.zip 为准。

新配置示例为 synthetic-corner-patch-shortcutminer-memory-v3.json 和 waterbirds-shortcutminer-memory-v3.json。这些仅显式声明构建版本；未实现或启用 confidence sequence，未增加配对样本预算，未降低阈值，未重新运行正式实验。

C001 的 qualitative metadata 缺失修复尚未实施，应作为单独、有测试和版本说明的修复处理，不借此改变统计策略。外部 Provider 重跑不能保证文字逐字相同；兼容性保证针对协议、选择记录、算法与历史产物。
