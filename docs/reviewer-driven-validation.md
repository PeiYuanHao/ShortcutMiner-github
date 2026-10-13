# Reviewer-driven validation requirements

Source: ICDM 2026 DM1230 reviews supplied by the author on 2026-10-04.
These reviews guide revision; they do not prescribe expected experimental results.

## Immediate core repair

Waterbirds must run through Identify, Executive, Quantitative, Qualitative,
EvidenceLedger and Summarize in the production LangGraph workflow. Quantitative
must produce its own training artifacts and ID/OOD predictions. A separate
standalone ResNet experiment is a model baseline, not a ShortcutMiner result.

The present Waterbirds protocol is metadata-assisted validation on a known
shortcut benchmark. It does not establish blind discovery from pixels. The
training-only discovery view excludes validation, ID and OOD records; validation
alone selects checkpoints. Official balanced test data is the OOD evaluation
partition. ID/OOD comparison is independent, not a paired causal intervention.
The predeclared Cramer's V > 0.5 rule is a paper heuristic, not independently
calibrated proof of shortcut reliance. Report all seeds and effect directions,
including small drops or contradictory results. Do not reproduce old paper
numbers unless newly measured.

## Required revision experiments not completed by a Waterbirds run

| Review issue | Required evidence | Current scope |
| --- | --- | --- |
| Agent contribution (R1-R4) | Fixed pipeline and single agent using identical tools, model budget and data; human audit effort | Pending controlled comparison; standalone classifier is insufficient |
| Component contribution (R1-R3) | Routing, memory, qualitative branch and template ablations with declared budgets | Pending; record unsupported ablations rather than simulate them |
| Unknown discovery (R1-R4) | Independently hidden cues, clean controls, and held-out datasets; freeze prompts/templates before disclosure | Pending; Waterbirds is a known benchmark |
| USPS hypothesis (R3) | Verify or refute the previously reported positional candidate | Pending; no confirmed finding claimed |
| Screening decisions (R1-R4) | Explicit semantic-versus-spurious decision protocol and ambiguous cases; independent raters and agreement where subjective | Pending; termination cannot establish absence of shortcuts |
| Thresholds (R1-R4) | Prespecified sensitivity analysis and calibration/held-out evaluation; false positives and recall | Pending; V > 0.5 retained as a disclosed heuristic |
| Backbone coverage (R1) | Repeated ID/OOD across model capacities and seeds | Completed core v7: ResNet18/50, three seeds; both are one model family |
| Reproducibility (R1-R4) | Prompts, tool templates, routing/promotion policy, source/config/data/checkpoint hashes, traces and failures | Local artifacts required; public release still pending |
| Practical value (R2-R3) | Wall time, token/compute costs, failure rate and manual dataset onboarding effort | Record available telemetry; unmeasured human effort must remain unreported |

## Writing boundaries

Distinguish dataset association, predictive reliance and causal evidence. Do not
assign causal claims to distributional accuracy changes. Position the current
implementation as an executable auditing workflow until controlled discovery
and baseline evidence support a stronger claim. Document the current LangGraph
and provider runtime rather than presenting the old OpenClaw/GPT implementation
as the environment of newly measured results. Correct inconsistent risk labels
and the reported 'shortcut hull learning' typo when revising the manuscript.
