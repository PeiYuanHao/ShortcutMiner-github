# 配对探测与早停 v2

保留五 Agent、Stage-1 / Stage-2、EvidenceLedger 和 Summarize。
提升只表示分配探测预算。旧 metric_gate 策略保留。
训练及配对探测由 Quantitative 的已授权训练工具执行。

## 冻结的候选协议

负责人确认 δ=0.02（准确率下降 2 个百分点）、总区间失败预算 α=0.05、
敏感性 δ=0.01/0.02/0.05。敏感性分析不更换主决策。
当前计算预算：每次运行最多 3 个候选，检查点 128/256/512 个笔画组，
按实际可用组数截断并去重。至多一半 ID 笔画组预留探测，其余用于独立确认。
这些是 owner-frozen 候选值，尚未经过跨数据集经验校准。
α 的解释以固定探测模型、独立 held-out 笔画组和规定检查点为条件，
不是所有模型或所有捷径类型的 5% 漏检保证。

## 分配与早停

Stage-1 执行廉价统计；V 只排序可执行候选，不再硬性否决弱关联。
控制层检查已编译计划：具有可执行训练依赖、配对干预步骤和调用者冻结的
adaptive_screening。未绑定配方的视觉候选不会借用其他候选的字段。
LLM 的 supported/置信度不触发提升。超出候选预算的记录为暂缓。

完整训练第一个 architecture/seed，当前为 ResNet18/42。
沿用原 epochs、训练早停和 validation 选点，不用短训无信号来否定模型依赖。
Validation accuracy 不优于 validation 多数类基线则暂缓解释；这是任务学习的
基本检查，不是模型质量的统计认证。

首个模型按检查点递增验证配对数据。同一 recovered digit intensity 的所有
着色版本属于一组：整组预留，每组随机取一个代表计算区间，防止重复计数。
保存原图、置换图、原像素重新保存的处理对照、全部 logits/预测及文件哈希。
逐像素检查 [R,G,B]→[B,R,G]、相同尺寸和坐标以及 min(R,G,B) 不变。
处理对照 logits 不一致时不能得出低影响结论。

设 D=原图预测正确−干预图预测正确，D∈{-1,0,1}，下降均值为
P(D=1)−P(D=-1)。两个概率使用 exact binomial Clopper–Pearson 区间。
每个检查点分得 α/(candidate_budget×实际检查点数)，两个概率再各分一半，
各概率的双侧再各分一半。差区间为 [L_loss−U_gain,U_loss−L_gain]。
以 Bonferroni 覆盖候选和所有规定检查点，不用重复查看的普通 bootstrap 决定早停。

- 上界 < δ：stop_low_effect，停止追加模型与独立最终确认预算。
- 下界 > δ：expand_models，训练其余冻结模型/种子，再在独立确认集验证。
- 区间跨越 δ：追加组；最后检查点仍不确定则 deferred_pair_budget_exhausted。
- 任务未学会、组数不足、处理对照失败：暂缓，不输出“不存在捷径”。

低影响只针对当前探测模型和当前干预，不代表其他模型也不依赖该线索。
有害、无影响和改善均保留符号；预测翻转率不能替代准确率下降。

## 复用、证据与恢复

相同运行内、相同训练/validation 数据内容、标签、后端和完整训练配方的
数字 NPZ checkpoint 可复用；不同运行或配置不能复用。
ToolRegistry 只对显式 adaptive 训练传递当前运行 shared_probe 数字缓存，
后端重新计算训练身份后读取对应项；拒绝符号链接和哈希变化。
各候选仍单独记录干预、抽样顺序、结果及血缘。

models/task/<model_id>/adaptive_screening.json 保存选项、决策、每次区间、
敏感性、全部预测/logits、处理对照与图像哈希。profiles.json、training_manifest.json
包含同一份哈希保护的决策。预留组的所有样本 ID 均从最终评估排除。

停止或暂缓时，OOD 工具只引用已经完成、同一配方的探测证据，
final_confirmation_executed=false，不伪装成独立最终确认。
扩大预算时，原 paired evaluator 在剩余独立样本上报告各模型/种子结果。
最终 bootstrap 区间用于描述，不驱动本次序贯预算决策。
科学结论仍由 EvidenceLedger 和真实 Summarize 生成。

当前模板支持 caller-frozen cycle_rgb。背景干预需要自己的保真和分组适配，
不能隐式套用数字笔画规则。

## Colorized-MNIST

新配置：configs/experiments/colorized-mnist-shortcutminer-adaptive-v2.json。
旧配置：configs/experiments/colorized-mnist-shortcutminer-formal.json 保留原 V>0.5。
入口默认 v2；使用新的 run ID，禁止原地更改已完成运行的策略。

```text
conda activate shortcutminer
python scripts/run_colorized_mnist_shortcutminer.py
python scripts/run_colorized_mnist_shortcutminer.py --live --run-id colorized-mnist-adaptive-v2-NEW-ID
```

第一条仅检查/准备数据。本轮未启动真实训练或 Provider。
后续论文比较新旧策略的 GPU 时间、提前停止比例以及相对于独立验证参考的漏检率
和误报率。两至三个种子不能估计所有模型的风险，δ=2 pp 也不宣称已跨数据集校准。
