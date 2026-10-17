# E4 reduced scope and protocol recovery

The user revised E4 after the original 28-cell matrix finished. The final selected
table contains E2-A and E2-B, each with Single-Agent, Fixed-Pipeline,
w/o Qualitative, w/o Quantitative and w/o Adaptive Routing. The eight completed
selected cells in study-v2 are reused unchanged. Only E2-A Single-Agent and
Fixed-Pipeline are rerun in a fresh recovery study. E2-C/E3 and the counterevidence
ablation are excluded; historical files and legacy variant code remain preserved.

Historical Full references are case-a-retest-r2 and case-b-v4-r2. They retain
their original source/configuration identities, candidate reports, assessments
and evidence. Reuse does not establish identical historical prompts, construction
operators or memory settings. Differences must be disclosed before attributing
historical-versus-current differences solely to an ablated component. No new
Full audit is launched by this recovery.

Both failed E2-A baselines exhausted schema retries on unauthorized metadata
fields in qualitative outputs. The opt-in metadata-contract-v2 revision explains
the existing allowlist from the current request at the multimodal structured-call
boundary. Visually observed properties belong in observations, not invented CSV
columns. Current sample/group reference requirements are also stated explicitly.
No output is rewritten or dropped; validators, retries, budgets, early stopping,
model training settings, thresholds and all evidence gates remain unchanged.
No cue identity, hidden truth, intervention recipe or desired outcome is supplied.
Legacy is still the default for existing launch commands.

## Descriptor codec recovery (v3 IO revision)

The instruction-only reduced-v3 attempts failed again. Their artifacts are
preserved. The subsequent descriptor-codec-v3 revision adds a bounded E4-only
encoding adapter at the actual multimodal validation boundary. It does not add
metadata permissions or remove validation. Exact descriptor aliases can become
the authorized bound column only when all current selected sample values match
uniquely, with at least two distinct values; label aliases are forbidden. A known
image descriptor supplied in the current inspection can otherwise be encoded as
observations, retaining the original text and all referenced samples. Existing
observation indexes, interpretation/confidence, counterevidence, status and
limitations stay unchanged. Unknown fields or unavailable samples are left for
the original validator to reject. Every final provider response, canonical
encoding and change receipt is persisted, including responses subsequently
rejected. No hidden truth is consulted by this adapter.

Development replays of the two previously failing public contexts are recorded
separately under runs/e4_ablation/codec-development. They are IO acceptance calls,
not additional paper audits, training or independent scientific replications.
The next two fresh audits use reduced-v4, the unchanged study-v2 profile/public
inputs and a new frozen source identity. Historical eight results are unchanged.

```powershell
python -X utf8 experiments/e4_ablation/scripts/prepare_reduced_study.py --previous runs/e4_ablation/study-v2 --study runs/e4_ablation/reduced-v4 --protocol-revision descriptor-codec-v3
python -X utf8 -u experiments/e4_ablation/scripts/run_matrix.py --study runs/e4_ablation/reduced-v4
```

New discovery starts from the same frozen public inputs, not old candidates.
The profile is byte-identical to study-v2. New source/archive hashes are recorded;
eight retained cells keep their old identities. Old failed attempts are retained
for execution-reliability and development-cost accounting. Completion is a runtime
outcome, not a guarantee of shortcut confirmation. Each new candidate still needs
post-run independent scoring; unknown truth remains unknown.

```powershell
python -X utf8 -m pytest -q experiments/e4_ablation/tests
python -X utf8 experiments/e4_ablation/scripts/prepare_reduced_study.py --previous runs/e4_ablation/study-v2 --study runs/e4_ablation/reduced-v3
python -X utf8 experiments/e4_ablation/scripts/consolidate_reduced_study.py --study runs/e4_ablation/reduced-v3
python -X utf8 -u experiments/e4_ablation/scripts/run_matrix.py --study runs/e4_ablation/reduced-v3
```

The runner executes only the two new cells sequentially and produces comparison
JSON/CSV/Markdown on completion. Consolidation references existing results without
rewriting them. It can be run read-only against prior runs while recovery is live.
