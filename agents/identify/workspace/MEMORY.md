# Stable memory policy

Do not store run facts. Each invocation relies only on its IdentifyRequest, IdentifyDiscoveryContext, and current task tool results.


## Literature mechanism memory v1

General hypotheses only; benchmark names and literature examples are not current-data evidence. Use `inspect_shortcut_methods` for cited sources, recipes, implementation status and limitations. Inspect independent mechanisms before selecting; unavailable backends must not be invented.
- global_photometry: Global brightness / exposure / contrast [geirhos2020,imagenetc2019,degrave2021].
- color_palette: Hue / channel ordering / saturation [irm2019].
- background_context: Background / context / environment [sagawa2020,groupdro2020].
- localized_artifact: Watermark / logo / text marker / corner patch [lapuschkin2019,geirhos2020].
- texture_shape: Texture / local statistics versus shape [geirhos2019].
- frequency: High-frequency signal / spectral sensitivity [hfc2020].
- geometry: Orientation / viewpoint / position / crop [objectnet2019,geirhos2020].
- encoding_quality: JPEG / blur / resolution / noise [imagenetc2019].
- source_style: Hospital / device / repository / scanner style [zech2018,degrave2021].
- accessory: Co-occurring object / treatment device / off-target marker [zech2018,degrave2021].
- demographic: Demographic or identity attribute correlated with target [groupdro2020].
- environment_shift: Location / camera / time / geographic domain [wilds2021].
- text_annotation: Negation / annotation artifacts / hypothesis-only bias [nli2018].
- text_heuristic: Lexical overlap / subsequence / constituent heuristics [hans2019].
- leakage: Duplicates / subject identity / split leakage [leakage2023].
Saliency needs sanity checks [saliency2018]; model dependence needs actual controlled evaluation. Full catalog: configs/knowledge/shortcut-literature-v1.json.

## Learned method memory

When enabled, Python freezes prior runs/knowledge/mechanisms into identify/learned-memory.json before discovery. The current payload or inspect_shortcut_methods supplies that immutable snapshot. These entries carry scoped model support, not universal verdicts. Unknown mechanisms may be discovered without any matching memory entry. New knowledge is written only after completed, backend-checked independent multi-model confirmation; current-run evidence cannot flow back into its own Identify stage.
