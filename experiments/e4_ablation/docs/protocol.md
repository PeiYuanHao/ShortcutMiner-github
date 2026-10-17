# E4 protocol v1

Historical protocol for the preserved study-v2 matrix. The user subsequently
reduced the reporting scope; see [current recovery protocol](reduced-recovery.md).
The original failures and results remain retained, not rewritten to fit that scope.

This protocol implements the user's E4-A and E4-B design without altering the
five-Agent defaults, scientific thresholds or previous run artifacts. Each of
four public inputs is independently audited by seven unique methods. The new
Full run is shared between the two comparison tables; historical reports are
not reused as candidate sources. E2-A/B/C inputs are the retained public images
and partitions. E3 uses the public inputs frozen for case-e3-representative-v1.
No human E3 adjudication exists (confirmed by the user). E3 truth-dependent
metrics are unavailable until independent adjudication is supplied.

## Methods

| Variant | Operational definition |
| --- | --- |
| full | Shared candidate-complete, representative-discovery, training-construction-review-v2 entry; five independent role contexts. |
| single_agent | One serialized controller identity with persistent shared memory across discovery, routing, numerical analysis, visual review and synthesis. Text and visual backends are tools of this controller. Typed phase adapters retain the same execution contracts and safety/evidence gates. It is a harness-controlled single-agent baseline, not an unconstrained code agent. |
| fixed_pipeline | Generic fixed descriptor ranking, one candidate per descriptor axis; descriptor measurement binding; fixed axis-to-operator lookup; visual safety review; fixed feasible statistical/training/evaluation/OOD order. Quantitative tools and report interpretation are real. No dataset-specific recipe is supplied. |
| without_qualitative | Disable Qualitative tasks and semantic construction review; Identify's visual discovery remains available. Numerical target-change checks remain. A null visual review means not performed, never passed. |
| without_quantitative | Disable Quantitative validation entirely, including measurement/construction planning and task-model training. Training-only discovery and qualitative synthesis remain. |
| without_adaptive_routing | Replace Executive validation routing with a frozen capability-checked order. Construction safety authorization and the existing resource-saving promotion/probe gates remain. |
| without_counterevidence | Disable counterexample_inspection and targeted construction-preview extremes, use 12 seeded-random previews. Retain all incidental contradictions, failed constructions and negative effects; keep primary paired OOD tests. |

Fixed order is association, subgroup evaluation, model training, ID evaluation,
paired OOD; unavailable checks are explicitly omitted. It does not bypass
feasibility or sequential early stopping. A fixed pipeline may defer a cue its
finite constructor vocabulary cannot express; that limitation is recorded.
Fixed-Pipeline uses a fixed twelve-per-class semantic observation schedule;
it does not ask a coverage agent to choose expansion. The same maximum allowance
is available to the other methods, which may stop earlier.

## Equal ceilings and source identity

All arms have the same text model, vision model, temperature and maximum output
tokens (8192), total text-call ceiling 200, vision-call ceiling 50, total tool
ceiling 150, five-candidate limit, three-construction-candidate cap and at most
three construction attempts per candidate. Per-role caps cannot disadvantage the
single controller: each role can access its channel's remaining common ceiling.
Calls and retries across discovery, construction and final validation share the
same channel budget; resume restores telemetry counters. Actual usage is measured.
These are call/output-token ceilings, not a promised equal number of billed tokens;
input/context costs, unknown usage and actual token totals are reported separately.

Training architectures, seeds, early stopping, delta, alpha, probe reservation and
confidence sequence are copied unchanged from image-audit.example.json. Two models
and three seeds remain conditional maxima. There is no new total audit timeout.
One independent discovery run per cell is planned initially: the three classifier
seeds do not supply between-audit replication or discovery-level error bars.
Dataset-major variant order is rotated across inputs. GPU arms run sequentially.

Fresh public-only copies contain exactly img_id,img_filename,y,split and images.
All arms use identical frozen rows and image hashes. No truth files, earlier
candidates or historical model weights are passed to any controller. Literature
knowledge is shared, learned run memory is excluded, and new memory promotion is
disabled for every arm to prevent cross-arm contamination. Source tree, source
archive, profile and inputs are hashed. A changed source tree stops the matrix
before another arm starts. Dirty source snapshots remain user-authorized.

## Measurement and interpretation

- Discovery Recall: unique planted mechanisms matched by a candidate divided by
  planted mechanisms; distinguish discovery from model-level verification.
- Verified Discovery Precision: independently validated true candidate mechanisms
  divided by all proposed candidate mechanisms. Duplicate and ambiguous matches
  require independent post-run matching; do not replace this with supported-report
  count. Also report precision among adjudicated candidates and adjudication coverage.
- False Positive Rate: specify the evaluation unit. On E2-C report false confirmation
  of the label-independent watermark; a single control yields a 0/1 outcome, not a
  precise population rate. Other natural cues have unknown truth. E2-C contains a
  watermark and is not a marker-free control.
- Validation Success Rate: candidates with at least one usable numerical or visual
  evidence record / all proposed candidates. Report model-tested count separately.
- Evidence completeness: per-candidate statistical, experimental, counterevidence
  and limitation coverage. Distinguish searching for counterevidence from actually
  observing it; a search that found none is not a fabricated contradiction.
- Costs: all provider attempts including retries and construction, known token
  totals, unknown-usage counts, tool calls and wall time; no overlapping stage sums.

Truth-dependent metrics remain null until independent post-run matching is frozen.
Unknown is never scored as false or zero. E3 has no complete truth or existing
human labels: report model sensitivities, assessments and unadjudicated counts.
No partial-support finding is silently upgraded to a confirmed nuisance shortcut.
The study uses previously inspected datasets and an adapted implementation;
describe it as a controlled comparison, not a new untouched blind benchmark.

## Windows commands

From the repository root in the shortcutminer Conda environment:

```text
python -X utf8 -m pytest -q experiments/e4_ablation/tests tests/test_representative_review.py tests/test_intervention_alignment.py
python -X utf8 experiments/e4_ablation/scripts/prepare_study.py --study runs/e4_ablation/study-v1
python -X utf8 -u experiments/e4_ablation/scripts/run_matrix.py --study runs/e4_ablation/study-v1
python -X utf8 experiments/e4_ablation/scripts/collect_results.py --study runs/e4_ablation/study-v1
python -X utf8 experiments/e4_ablation/scripts/score_adjudication.py --study runs/e4_ablation/study-v1 --adjudication PRIVATE_FROZEN_JUDGMENTS.json
```

Progress and results are English Markdown, JSON and CSV in the study folder.
Failures remain in the matrix and are not counted as scientific negatives.
Resuming failed arms is explicit with --retry-failed and requires identical
source/config/input identity. Never overwrite an earlier arm to tune its result.
