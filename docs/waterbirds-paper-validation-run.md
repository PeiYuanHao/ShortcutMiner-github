# Waterbirds 论文验证：当前结果入口

2026-10-05 用户要求保留最新五 Agent＋本地构建成对 OOD 的结果，并清理此前结果。
旧独立模型验证结果已删除，其训练入口与配置保留用于实现参考。
当前保留运行：`runs/waterbirds-shortcutminer-paired-20261005-v1`。

- 结果与边界：该运行的 `results-overview.md`。
- 原始 Summarize / EvidenceLedger 报告：`audit_report.md`。
- 配对、预测和模型身份独立复核：`paired-verification.json`。
- 配置：`configs/experiments/waterbirds-shortcutminer-background-formal.json`。
- 五 Agent 执行说明：`docs/waterbirds-shortcutminer-core-run.md`。
- 本地分割与 OOD 构建：`docs/local-background-ood.md`。

Windows conda shortcutminer，Python 3.11.16，torch 2.9.1+cu128，
torchvision 0.24.1+cu128，Transformers 5.14.1；不使用容器。
本次为 3836 train / 1199 validation / 959 ID，成功构建 954 对背景中性化样本。
已有 capacity 与环境 acceptance 记录保留，不作为正式论文结果。
模型 baselines、消融、隐藏 metadata 的盲发现和其他数据集仍需按受控协议另行验证。
