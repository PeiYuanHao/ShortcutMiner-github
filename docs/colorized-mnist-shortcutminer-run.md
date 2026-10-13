# Colorized MNIST：原始数据的 ShortcutMiner 实验

入口现默认使用 [配对探测与早停 v2](adaptive-paired-early-stop.md)：
configs/experiments/colorized-mnist-shortcutminer-adaptive-v2.json。
下述 V>0.5 配置和已完成运行属于旧策略，保留作为对照，不原地改变。
新策略先执行一个模型/种子的有预算配对初筛，再决定停止或扩展完整矩阵。
本次代码修改不启动新真实实验。

来源：`E:/小龙虾/colorized-MNIST-master`，原 training 8835、testing 10000。
原图不改色、不新增颜色与标签的关联、不修改晋升阈值。
标签取自 0～9 类别目录；颜色文件名前缀与全部 RGB 像素一致性检查通过。
颜色作为公开 metadata，是 metadata-assisted 运行，不宣称盲发现。

输入协议在 `configs/experiments/colorized-mnist-shortcutminer-formal.json` 冻结。
全部文件内容、恢复的 digit intensity 及原来源都有 SHA-256。
逐像素 canonical composition 为 `RGB = max(digit_intensity, color_RGB)`。
以 digit intensity SHA-256 为分组单位隔离分区；与原 test 相同内容的
training 记录从训练侧排除，原 testing 保留。
原 training 的 eligible groups 预留 10% ID，其余 20% 用于 validation，seed 42/43。
同一笔画内容的重复图像绝不跨 train / validation / ID / test。
发现与 Stage-1 只读 train metadata；输入视图位于 `runs/colorized-mnist-shortcutminer-input-v1`。

若现有流程晋升候选到 Stage-2，Quantitative 工具执行当前运行内的新训练：
ResNet18/ResNet50 × seeds 42/43/44，ImageNet 初始化，batch 32，SGD lr 0.001，
momentum 0.9，weight decay 0.0001，AMP，4 workers，24 CPU threads。
最多 100 epochs、最低 10、patience 10，validation accuracy 选择检查点。
数字的训练和推理都只做完整 Resize(224,224)、tensor、ImageNet normalization；
不用翻转或随机裁剪。没有新增依赖，不使用 Waterbirds 或 baseline 检查点。

颜色干预为 frozen RGB channel cycling `[R,G,B] -> [B,R,G]`，
红→绿、绿→蓝、蓝→红。保留空间坐标和每个像素的 recovered digit intensity。
在 held-out ID 上生成一对一无损 PNG，逐模型保存两侧 logits、预测、准确率、
带符号的下降和翻转率，三种子均值 ± 样本标准差。
这不是把原 testing 随意命名为 OOD，也不需要使用 bird segmentation。
是否符合完整语义保真仍需检查；channel permutation 本身不会改变恢复的笔画强度。

本实验遵循现有 Stage-1/Stage-2 规则：框架可支持、否定或保持不确定。
没有候选晋升时，Stage-2 跳过必须如实报告，不伪造训练或模型依赖结果。
数据关联与干预敏感性是不同的证据；不提前向 Agents 提供预检统计或期望结论。
识别和定性抽样上限各 20，Stage-2 定性按数字类别平衡抽样。
定性向 VLM 提交一张编号联系表，包含同一批已检查的 20 张图片；
数字以 nearest-neighbor 放大，不改色。源图哈希、样本 ID 和 cell 映射单独保存，
ModelInvocation 记录实际提交的一张合成图片及其 SHA-256，不冒充提交 20 张。
该传输模式只由此数据集入口启用，Waterbirds 保持默认单图提交。

```text
conda activate shortcutminer
python scripts/run_colorized_mnist_shortcutminer.py
python scripts/run_colorized_mnist_shortcutminer.py --live
```

Credentials 仅在 live 进程内从 ignored `.env.formal` 加载。
Windows native conda；没有 Docker，也不设置整体实验 timeout。
训练 tool 不设整体 timeout；ID/OOD tools 各 1800 秒，其他 tools 300 秒，
现有 Provider timeout/retry 保留。运行保存配置/输入/源码 archive 身份。
失败恢复须保持同一配置、输入和 run ID：

```text
python scripts/run_colorized_mnist_shortcutminer.py --live --resume --run-id EXISTING_RUN_ID
```

已完成的 Waterbirds 结果和 archive 保持原样。新增适配后的源树属于新执行快照。

2026-10-05 实际运行 `runs/colorized-mnist-shortcutminer-20261005-v1` 已 COMPLETED。
Identify、Executive、Quantitative、Qualitative、Summarize 均有真实成功调用，
5 个候选均有定性结果，每项检查 20 张类别平衡样本。
Summarize 原始输出为 `summarize/result.json`；`audit_report.md` 是其规范报告呈现。
独立验收 `completed-verification.json` 核对源码归档、配置和输入哈希，并重算统计。
训练分区 6359 条记录：chi-square=24.7569838021、df=18、p=0.1317578568、
Cramér's V=0.0441204015。C002 未达到冻结 V>0.5 规则，其他候选证据不足。
因此本次 Stage-2 训练和配对 OOD **未执行**，不能宣称已证明模型依赖或不存在捷径。
ResNet 配置及颜色干预配方只是条件执行方案，本次没有生成这类实验结果。

为解决真实 VLM 响应耗时超过非流式等待期限的问题，此入口使用流式 HTTP 接收，
仍保留 60 秒连接/无数据读取超时；仅接收完成标记和结束原因齐全的完整响应。
原始证据保存在 input.json，重复描述通过可逆字段路径表传输，所有值和记录保留。
联系表和实际发送字段路径表纳入 invocation payload hash。
协议修正明确既有跨组比较规则，没有放宽校验或改变 scientific thresholds。
参考：[Qwen 流式输出官方文档](https://www.alibabacloud.com/help/zh/model-studio/stream)。
修复后完整回归为 955 passed、11 skipped；最后提示说明变更另有 13 项相关测试通过。
