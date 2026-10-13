# 文献驱动的捷径发现与验证方法库

检索日期：2026-10-08。范围：有针对性的原始论文检索，并非穷尽文献。方法库不含当前数据集的隐藏规则。

论文中的实例只用来拓展候选机制。当前数据证据决定候选与参数；验证失败不能倒推修改阈值。对文献方法的工程改编已在每项限制中注明。

| 类型 | 检查与验证方法 | 当前执行状态 | 文献 |
|---|---|---|---|
| Global brightness / exposure / contrast | Training-calibrated Y-mean normalization or reflection; no-op paired control; same frozen model and reserved confirmation. | executable_general_rgb | [geirhos2020](https://arxiv.org/abs/2004.07780), [imagenetc2019](https://arxiv.org/abs/1903.12261), [degrave2021](https://www.nature.com/articles/s42256-021-00338-7) |
| Hue / channel ordering / saturation | Authorized palette permutation, grayscale/desaturation or channel scaling; paired evaluation with unchanged geometry. | executable_general_rgb | [irm2019](https://arxiv.org/abs/1907.02893) |
| Background / context / environment | Frozen foreground segmentation and background replacement or approved scene swaps; foreground-only controls. | executable_with_segmentation_prerequisites | [sagawa2020](https://proceedings.mlr.press/v119/sagawa20a.html), [groupdro2020](https://arxiv.org/abs/1911.08731) |
| Watermark / logo / text marker / corner patch | Agent-localized mask, replacement, matched patch removal or position randomization; keep processing artifacts and content-loss controls. | executable_with_localization_prerequisites | [lapuschkin2019](https://www.nature.com/articles/s41467-019-08987-4), [geirhos2020](https://arxiv.org/abs/2004.07780) |
| Texture / local statistics versus shape | Blur or frequency sensitivity is a diagnostic; faithful shape-texture cue-conflict / stylization needs approved construction and semantic checks. | diagnostic_executable_stylization_not_implemented | [geirhos2019](https://arxiv.org/abs/1811.12231) |
| High-frequency signal / spectral sensitivity | Bounded Fourier low/high-pass filtering and blur with paired no-op controls. | executable_general_rgb | [hfc2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Frequency_Component_Helps_Explain_the_Generalization_of_Convolutional_Neural_Networks_CVPR_2020_paper.html) |
| Orientation / viewpoint / position / crop | Reflect-padded rotation or bounded spatial intervention; natural viewpoint transfer requires a separately approved dataset. | rotation_executable_3d_viewpoint_not_implemented | [objectnet2019](https://papers.nips.cc/paper_files/paper/2019/hash/97af07a14cacba681feacf3012730892-Abstract.html), [geirhos2020](https://arxiv.org/abs/2004.07780) |
| JPEG / blur / resolution / noise | Recompression, blur, resolution and noise paired stress tests; report exact persistence round-trips. | executable_general_rgb | [imagenetc2019](https://arxiv.org/abs/1903.12261) |
| Hospital / device / repository / scanner style | Approved source subgroup/external validation; visual style perturbations are proxies; true acquisition-source swapping needs authorized data. | observational_tools_available_external_data_required | [zech2018](https://doi.org/10.1371/journal.pmed.1002683), [degrave2021](https://www.nature.com/articles/s42256-021-00338-7) |
| Co-occurring object / treatment device / off-target marker | Mask/background intervention if localization is sound; domain expert semantic review required for medical conclusions. | executable_with_localization_and_domain_review | [zech2018](https://doi.org/10.1371/journal.pmed.1002683), [degrave2021](https://www.nature.com/articles/s42256-021-00338-7) |
| Demographic or identity attribute correlated with target | Approved subgroup/worst-group analysis; attribute counterfactual edits require validated domain-specific tools. | subgroup_tools_available_attribute_editing_not_implemented | [groupdro2020](https://arxiv.org/abs/1911.08731) |
| Location / camera / time / geographic domain | Independent approved environment evaluation and worst-group metrics; source-group-disjoint splits when available. | observational_tools_available_external_data_required | [wilds2021](https://proceedings.mlr.press/v139/koh21a.html) |
| Negation / annotation artifacts / hypothesis-only bias | Hypothesis-only or cue-only baselines and hard subsets; text backend must be supplied. | method_memory_only_no_text_training_backend | [nli2018](https://aclanthology.org/N18-2017/) |
| Lexical overlap / subsequence / constituent heuristics | HANS-style controlled minimal pairs and syntactic generalization; domain text generator/backend required. | method_memory_only_no_text_counterfactual_backend | [hans2019](https://aclanthology.org/P19-1334/) |
| Duplicates / subject identity / split leakage | Duplicate/subject-disjoint checks; correctly isolated retraining if leakage is established. | exact_hash_checks_available_subject_metadata_required | [leakage2023](https://arxiv.org/abs/2207.07048) |

## 完整性与结论边界

可执行图像操作已连接到同一冻结模型的配对早停入口。SAM/ViTMatte 背景路线保留原有授权与质量检查。真实风格迁移、3D视角、文本模型、敏感属性编辑仍需要相应后端；不能以方法库条目冒充执行结果。

全图像素变化率是诊断信息，不能独立判定语义破坏。Qualitative 必须看到原图/变换图配对；Executive 结合图像评审和结构检查作出范围明确的授权。视觉评审不是语义证明。

显著性图不能单独证明依赖；参见 [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)。独立留出配对和资源早停策略保持既有冻结值。

## 来源清单

- 2020 [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780)
- 2019 [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)
- 2019 [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
- 2020 [An Investigation of Why Overparameterization Exacerbates Spurious Correlations](https://proceedings.mlr.press/v119/sagawa20a.html)
- 2020 [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731)
- 2019 [Unmasking Clever Hans predictors and assessing what machines really learn](https://www.nature.com/articles/s41467-019-08987-4)
- 2021 [AI for radiographic COVID-19 detection selects shortcuts over signal](https://www.nature.com/articles/s42256-021-00338-7)
- 2018 [Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs](https://doi.org/10.1371/journal.pmed.1002683)
- 2021 [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://proceedings.mlr.press/v139/koh21a.html)
- 2019 [ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models](https://papers.nips.cc/paper_files/paper/2019/hash/97af07a14cacba681feacf3012730892-Abstract.html)
- 2020 [High-Frequency Component Helps Explain the Generalization of Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Frequency_Component_Helps_Explain_the_Generalization_of_Convolutional_Neural_Networks_CVPR_2020_paper.html)
- 2019 [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261)
- 2018 [Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/)
- 2019 [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://aclanthology.org/P19-1334/)
- 2018 [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)
- 2023 [Leakage and the Reproducibility Crisis in ML-based Science](https://arxiv.org/abs/2207.07048)

## 接入与使用

Identify 的常驻 memory 提供机制索引，`inspect_shortcut_methods` 返回完整引文、实际能力和前置条件。训练图像中的五组独立证据轴同时进入 Identify payload；这是覆盖提示，不会强制生成亮度候选，也不会把统计关联升级成模型依赖结论。样本 ID、路径和 split 不允许被绑定为科学测量字段。

新增 `calibrate_luminance` 工具只允许 Quantitative 使用，训练目录、CSV、列和 split 由 Python 提供，Agent 只能选择模式与拟合策略。YCbCr Y 的新增均值/标准差使用 0–255 单位并标记 `ycbcr_y_v1`；旧 intensity 指标仍为 0–1，旧已保存报告不重新计算。

`infrastructure/experiments/run_autonomous_image_audit.py` 是显式启用的通用入口。它先运行公开 train-only 的五 Agent 发现，冻结候选；Quantitative 选择构造，Python 拟合参数并保存训练配对图像，Qualitative 实际看图，Executive 审核。接受之后才打开 model 输入的 val/id 图像，派生候选绑定描述符，继续通过原五 Agent workflow 执行训练和配对验证，最终报告仍由 Summarize 产生。模型、种子、训练策略、promotion policy、δ、alpha、敏感性和区间方法从既有冻结 profile 读取，不按本次结果修改。源快照、输入和构造文件哈希受 resume 检查。不会预生成评估 OOD；实际 OOD 在授权的定量工具中构造。

下面是 PowerShell 的调用形式，替换公共目录、既有冻结协议和公开任务说明。该命令会调用真实 Provider；本次修改与测试没有运行它。

```powershell
python infrastructure/experiments/run_autonomous_image_audit.py --input "<公开训练目录>" --training-input "<公开train-val-id目录>" --profile "<冻结实验协议.json>" --run-id "<新的run-id>" --task "<公开任务说明>" --validation
```

仅检查规划时将 `--validation` 换成 `--proposal-only`：它仍调用发现/规划 Provider，但不打开留出图像、不训练任务模型。后续 `--validation` 使用同一 run-id 和冻结源码继续执行。

公开 CSV 必须只有 `img_id,img_filename,y,split`；发现目录必须全为 train，模型目录为 train/val/id，二者训练记录与图像必须一致。该入口显式用于开发执行，默认不宣称 confirmatory blind evidence；用过的留出集不能重新变成首次盲式验证。旧实验入口与版本化 OOD 选择协议保留，旧结果文件不覆盖。

通用旋转、频域过滤、亮度调整、噪声、压缩等都是有限范围的诊断。处理对照、实际模型预测、原样本配对和独立确认保留；类别语义与物理实例独立性不能从像素唯一性或看图结果自动认证。真实背景 OOD 仍走原先带 SAM/ViTMatte 前置条件的路径，通用入口不会下载或猜测分割模型。

## Identify 自主检索与实验后学习

新增 `search_shortcut_literature`，由 Identify 自己从训练观察选择通用英文机制关键词，直接检索 arXiv 的论文题名、作者、日期与摘要节选。不是固定题库，也不要求新候选属于已知 memory 条目。未知机制仍可使用原协议的其他类别，并向 Executive 提出相应验证需求；没有验证后端时必须明确记录缺口。检索结果不作为当前数据集的 evidence_source，不执行论文中的代码，不宣称只读摘要就审查了完整论文。

通用新入口已启用 `configs/knowledge/research-policy-v1.json`：每次发现最多三次检索、每次最多三篇、请求时限十五秒；实际 API 请求遵守三秒间隔，缓存成功/失败回执，resume 不补发同一查询或重置次数。依据 [arXiv API 官方说明](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md)。查询不接受数字、路径、URL、密钥或隐藏规则，拒绝当前数据集标识；常规候选仍须有当前训练数据证据。网络未启用的旧流程和离线流程保持原策略。

其他 Runner 配置可显式设置：

```json
{
  "identify_config": {
    "research_policy": {
      "enabled": true,
      "max_queries": 3,
      "max_results": 3,
      "timeout_seconds": 15,
      "include_learned": true
    }
  },
  "metadata": {
    "knowledge_policy": {"enabled": true, "max_entries_per_run": 3}
  }
}
```

方法记忆与联网独立：`enabled:false, include_learned:true` 可以离线读取已冻结的旧记忆，不发送查询。旧配置默认不开启网络或实验后写入。两阶段新入口仅在初始发现时启用检索和旧记忆；验证阶段回放冻结的 Identify 输出，不用当前结果重新挑选候选。

实验完成后，Python 执行额外的知识回写检查，不改变五 Agent workflow、Ledger、Summarize 报告或实验阈值。当前自动认证入口支持实际 torchvision 多架构/多种子矩阵：

- 已保存 COMPLETED 状态和一致的 Summarize `supported` 结论。
- 对应成功的定量 ToolResult、正式配对测量、模型 manifest 与权重哈希。
- 冻结计划中的全部架构/种子均完成验证，至少两种架构；不能用单个 pilot 或模型未学会的结果。
- 各模型配对效果和已有置信区间下界均超过该次原先冻结的 δ；不新增或回调科学阈值。
- 确认样本不与早停 probe 重叠；不把关联、发现、预算晋级或显著性图当认证。

通过后由 Identify 作一次单独的概括，将新机制与验证方法转换成通用英文记忆。禁止数据集名称、标签对应关系、植入比例/数值、路径和当前结果数值。保存为 `runs/knowledge/mechanisms/<mechanism-id>.json`；详细证据与哈希保存在 `runs/knowledge/provenance/`，当前实验记录在 `<run>/knowledge/learning-result.json`。已有机制定义不覆盖，重复成功执行可增加独立来源证明。

每次新 Identify 启动前，旧方法库冻结到 `<run>/identify/learned-memory.json`，其哈希进入 Identify 输入；`inspect_shortcut_methods` 同时提供文献方法库和这份旧记忆。完成本次实验后新增的条目只影响未来快照，resume 使用原快照，避免测试结果回流和盲式实验污染。不会自动修改过去的报告或补填过去的认证。

记忆状态为 `model_supported_scoped`：表示特定训练模型与干预下的支持，不表示普遍因果关系，也不表示国际上首次发现。语义改变、物理实例独立性与区间适用条件继续作为限制保留。未知后端、缺少明确 δ、区间跨界、部分训练或提炼失败会记为 deferred，不会写成已认证条目。
