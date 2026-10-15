# ShortcutMiner manuscript revision highlights

Recorded: 2026-10-08. The user explicitly requested preserving these changes for the final manuscript revision. This is a contribution and evidence checklist, not a declaration that every capability has already been demonstrated in final paper experiments.

## Changes to discuss in the manuscript

| Highlight | What changed and why it matters | Manuscript location / evidence | Claim boundary |
| --- | --- | --- | --- |
| Complete framework-owned validation | Discovery, qualitative review, Executive authorization, Quantitative model training and paired validation, and Summarize reporting belong to the five-Agent workflow. Standalone classifier experiments must be identified separately as baselines. | System and evaluation; `reviewer-driven-validation.md`, `e2-autonomous-intervention.md` | A baseline result is not evidence that the complete ShortcutMiner workflow succeeded. Cite actual framework run artifacts. |
| Blind discovery and Agent-owned intervention | The E2 generator implants the shortcut and saves private truth; it does not specify or pre-generate OOD. Agents discover candidates from public inputs and propose candidate-bound interventions. Hidden truth is reserved for external post-run scoring. | E2 protocol; `e2-autonomous-intervention.md` | Keep private truth out of prompts, learned method memory, and same-experiment feedback. Known-shortcut rediscovery and blind discovery are separate evaluation settings. |
| Literature-grounded, broader hypothesis space | A targeted catalog covers 16 primary papers and 15 mechanism families. Independent photometry, chromaticity, texture/structure, spatial, and acquisition/encoding evidence axes reduce domination by near-duplicate descriptors. | Identify method; `shortcut-literature-and-validation.md`, `../configs/knowledge/shortcut-literature-v1.json` | Targeted coverage is not an exhaustive survey. A catalog entry is not evidence that this dataset contains that shortcut. |
| Autonomous literature retrieval | Identify can choose generic mechanism queries and retrieve public arXiv metadata/abstracts within a frozen query budget. Search has privacy restrictions, cached provenance, resume-consistent budgets, and explicit failure handling. Unknown candidate mechanisms remain allowed. | Identify tools and reproducibility; `../shortcutminer/experiment/knowledge_research.py`, `../configs/knowledge/research-policy-v1.json` | Retrieved text is untrusted background, not current-dataset scientific evidence. Search benefit requires an ablation; search failure does not establish shortcut absence. |
| Training-only intervention calibration | Quantitative chooses a candidate and transformation strategy; parameters such as luminance targets are fitted label-blind from pooled training images, then frozen before held-out validation. | Quantitative construction; `../shortcutminer/experiment/luminance_calibration.py`, `../shortcutminer/experiment/autonomous_construction.py` | Dataset implantation parameters and held-out model outcomes must not determine the validation transformation. Parameter fitting does not itself certify semantic preservation. |
| Real paired visual review | Qualitative receives actual paired contact-sheet images. Executive considers visual review and pixel diagnostics before permitting validation. Global image change is no longer rejected solely because many pixels changed; unsuccessful visual review cannot be treated as acceptance. | Construction governance; `../shortcutminer/experiment/autonomous_construction.py` | Report content preservation and artifacts with their actual review evidence. Pixel statistics alone are not semantic certification. |
| More executable validation mechanisms | Generic RGB validation now includes luminance operations, rotation, frequency filtering, and existing noise/blur/compression/resolution/chromatic operations. Earlier local SAM/ViTMatte background construction remains a separate mechanism with prerequisites. | Tool implementation; `shortcut-literature-and-validation.md`, `../shortcutminer/intervention/capabilities.py` | Generic corruption-induced accuracy loss is not automatically causal shortcut proof. Some literature methods remain documented proposals without executable backends. |
| Resource-aware sequential validation | Cheap association prioritizes candidates rather than acting as a hard negative veto. Paired model evidence supports early stopping, expansion, or deferral. The user selected a 2 percentage-point meaningful-drop candidate, 1/2/5-point sensitivity, and a 5% error budget. An opt-in paired confidence sequence preserves the legacy interval path. | Statistical method; `adaptive-paired-early-stop.md`, `paired-confidence-sequence-v1.md` | These choices are predeclared policy, not objectively learned optimal thresholds. State interval assumptions and multiplicity scope. Speed/resource claims need actual telemetry and matched-budget comparisons. |
| Evidence-gated method learning | After durable completion and a saved matching SummaryReport, a strict gate checks canonical Quantitative results, tool provenance, actual frozen model/seed coverage, full paired evidence, uncertainty bounds, disjoint probe/final IDs, and artifact hashes. Identify then abstracts a reusable method. | Memory update method; `../shortcutminer/experiment/learned_shortcuts.py` | Entries are model-supported and scoped. Local memory novelty is not global scientific novelty; individual model intervals do not add an unimplemented simultaneous confidence guarantee. Incomplete evidence is deferred. |
| Future-only, auditable memory reuse | Generalized mechanisms and separate corroborating provenance are stored under ignored `runs/knowledge`. Each new run freezes a method-memory snapshot. Existing snapshots and historical reports are not rewritten. | Reproducibility and continual method reuse; `../shortcutminer/experiment/learned_shortcuts.py` | Do not claim measured learning gains until independent future-run evaluation. Public method memory excludes dataset identities, class mappings, hidden ratios, and source-specific results. |
| Backward compatibility and reproducibility | New tools and statistical modes are opt-in where required; source/config/input/weight hashes, receipts, query caches, and resume checks preserve experiment identity. New generic entry uses training-only discovery followed by construction and the existing five-Agent validation workflow. | Appendix; `../infrastructure/experiments/run_autonomous_image_audit.py`, `paired-confidence-sequence-v1.md` | Do not retroactively certify old runs with newer code. Two ResNet variants represent two architectures within one model family, not cross-family diversity. Pixel uniqueness does not establish physical-instance independence. |

## Evidence available versus still needed

- Retained complete E3 development recheck:
  `runs/e3-representative-20261014/case-e3-representative-v1`. The actual Summarize
  Agent gives the new texture/edge C003 **partially_supported**. Conditional
  expansion completed ResNet18/ResNet50 at seeds 42/43/44; 2,169 final pairs are
  disjoint from reserved probe IDs. Architecture mean drops are 7.21 ± 0.23 pp
  and 6.61 ± 0.51 pp (seed SD, shared samples). This demonstrates scoped
  intervention dependence, not clean nuisance-texture causality or independent
  blind confirmation. See the retained execution-review and corrected readable
  report before quoting numbers. The raw SummaryReport also inherits some
  discovery-stage limitations; completed model/prediction facts must be checked
  against the actual final artifacts rather than those stale phase statements.

- Added opt-in class-representative-v1 observation (full bounded training numeric
  scan and Agent-budgeted 4/8/12 semantic images per class), plus training-only
  construction-review-v2 with random/extreme paired previews, explicit mechanism
  scope, artifact diagnostics and damage/unsupported-operation rejection. A
  separate bounded color-ratio operator preserves float RGB luminance and
  compresses chroma into gamut. This is implementation evidence, not a confirmed
  shortcut or a claim of improved discovery rates. Scientific gates and adaptive
  model budgets remain unchanged. See
  `representative-discovery-and-construction-review.md` for limits and opt-in use.

- Added opt-in primary measurement/intervention alignment after E3 exposed a
  chromaticity candidate receiving a luminance operation. The new check binds
  the frozen descriptor axis to an eligible operation and measures actual
  training-preview target changes before training, with bounded Agent revision
  and unchanged legacy callers. See `intervention-measurement-alignment.md`.
  This is an execution-quality improvement, not evidence that new shortcuts
  were confirmed; old E3 results remain inconclusive and retain their artifacts.
  A training-fitted channel-ratio neutralization/log-reflection constructor now
  closes the corresponding chromaticity calibration gap. Real E3 training-only
  acceptance narrowed the measured ratio range and passed scoped paired visual
  review; it did not train a new model or certify shortcut reliance.

- Literature/validation changes were checked with related offline regression suites: 297 unique relevant passing tests and two Windows symlink skips across the recorded batches. Retrieval/learning changes had a broad 141-test passing batch plus subsequent targeted checks. Counts overlap across batches and must not be added into a single claimed suite total.
- A live public arXiv retrieval succeeded. These checks did not run real dataset audits or real text/VLM providers and are not E2 scientific results.
- The latest EuroSAT repairs and new learning capability still require an actual framework run and independent outcome review. Previous erroneous reports are not automatically repaired by code changes.
- Final manuscript numbers must come from retained run artifacts, model predictions, paired uncertainty records, SummaryReport, configurations, and source snapshots. Verify their current locations and identities before citation.

## Evaluation additions to prepare

1. Separate E1 known-shortcut rediscovery from E2 blind discovery; score against hidden truth only after the completed audit.
2. Compare search enabled/disabled and prior learned memory enabled/disabled under matched candidate, model, and resource budgets.
3. Evaluate previously unseen mechanisms on independent data; report missed discoveries, false positives, duplicate candidates, unsupported constructions, and deferrals.
4. Compare the complete framework with appropriate fixed-pipeline/single-Agent and classifier baselines; keep baseline conclusions separate.
5. Report wall time, model-training count, GPU/resource use, provider calls, and early-stop savings together with accuracy and paired evidence.
6. Include threshold sensitivity and the assumptions of the selected sequential interval method. Assess semantic preservation with actual paired reviews and, where feasible, independent raters.

## Writing boundaries to preserve

The main narrative can connect autonomous discovery, Agent-owned counterfactual construction, economical model-level verification, and evidence-gated reusable memory. Distinguish implementation improvements from demonstrated research findings. Association, model dependence, and causal interpretation require different evidence. Do not describe a new local memory entry as the first discovery in the literature. Do not describe documented but unsupported methods as executable tools. Do not claim the subjective-threshold concern is eliminated: transparent predeclaration, sensitivity, and sequential uncertainty address it only within their stated scope.

The runtime remains Windows local Conda `shortcutminer`, not Docker/Linux. Credentials are loaded from local `.env.formal` and must not appear in manuscript artifacts or memory. No automatic commit is authorized. This record does not start experiments or alter scientific thresholds.
