# Waterbirds：当前保留的五 Agent 实验

当前结果：`runs/waterbirds-shortcutminer-paired-20261005-v1`。
2026-10-05 完成真实 Identify、Executive、Quantitative、Qualitative、Summarize。
原始报告、模型检查点、源码快照、逐样本预测与 OOD 图像全部保留。

| 模型 | 原图准确率 % | 背景干预 OOD 准确率 % | 下降 pp |
|---|---:|---:|---:|
| ResNet18 | 96.75 ± 0.18 | 95.91 ± 0.73 | 0.84 ± 0.58 |
| ResNet50 | 97.80 ± 0.10 | 96.89 ± 0.40 | 0.91 ± 0.30 |

种子 42、43、44 的均值 ± 样本标准差，两个模型均为本轮新训练。
训练 3836、验证 1199、预留 ID 959；只用验证集选择检查点。
本地 Grounding DINO → SAM → ViTMatte 构建背景中性化图像：954 对成功，5 张失败。
原图与干预图逐样本配对，成功样本的预测保护区域像素保持一致。
`paired-verification.json` 独立复核为 PASS；原始五 Agent 报告是 `audit_report.md`。
完整数字、方向与解释见该运行的 `results-overview.md`。

Summarize 对 C001/place 给出 partially_supported。预测分割与标签保真仍需独立核验。
本条件为单一灰色背景；发现为 train metadata-assisted。
受控 baselines、消融与盲发现仍待完成，整篇论文验证尚未全部完成。

## Windows 本地启动

CMD 或 PowerShell，使用 conda `shortcutminer`。Credentials 仅从 ignored
`.env.formal` 在进程内加载。当前配置：
`configs/experiments/waterbirds-shortcutminer-background-formal.json`。

```text
conda activate shortcutminer
python scripts/run_waterbirds_shortcutminer.py
python scripts/run_waterbirds_shortcutminer.py --live --config configs/experiments/waterbirds-shortcutminer-background-formal.json
```

第一条 Python 命令验证并准备输入，不调用 Provider；第二条创建新的正式运行。
训练没有整体 tool timeout；ID/OOD tools 各 1800 秒，其他 tools 300 秒。
已有 Provider timeout/retry 保留。ResNet18/50、三个种子、batch 32、SGD lr 0.001、
momentum 0.9、weight decay 0.0001、AMP、4 workers、24 PyTorch CPU threads。
最多 100 epochs、最低 10 epochs、patience 10，验证准确率同分保留较早检查点。

保留 `runs/waterbirds-shortcutminer-input-v1`：它是当前结果的输入视图。
源数据仍在 `E:/ShortcutMiner-Experiments/waterbirds`。执行 archive 和 receipt
固定本轮实际执行身份；清理后的文档源树属于新快照，不能替换原执行身份。

失败的新运行可在保持原配置和数据的前提下恢复：

```text
python scripts/run_waterbirds_shortcutminer.py --live --resume --run-id YOUR_EXISTING_RUN_ID
```

已完成运行不能 resume。新运行自动生成新的时间标识。
2026-10-05 用户授权清理旧结果：旧试跑、旧独立模型结果及 formal-v1～v7 已删除。
删除清单与当前结果文件保留校验见当前运行的 `cleanup-receipt.json`。
