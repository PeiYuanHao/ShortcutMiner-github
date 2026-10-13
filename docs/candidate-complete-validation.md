# Candidate-complete-v1 validation

This is an explicit opt-in execution policy. `run_autonomous_image_audit.py`,
`run_blind_spatial_image_audit.py`, existing configuration defaults, protocol
serialization, thresholds and archived experiment reports remain unchanged.
Use a new run ID/root for the new policy; never resume an old-source run under it.

Quantitative first chooses one primary registered image descriptor or ratio for
each discovered candidate, before statistical tests or held-out access. The
validated plan covers all frozen candidate IDs exactly once. Arbitrary code,
labels, paths, split IDs and private truth are not measurements. A metadata
hypothesis without an authorized metadata field cannot be silently replaced by
an image proxy. Zero-denominator ratios are missing, not fabricated zero values.

Training hashes and measurement plans are frozen and checked on resume. Constant
or unavailable measurements have explicit structural dispositions, not negative
shortcut verdicts. All usable fields receive candidate-specific bindings, even
if no OOD construction can be accepted. Shared descriptor extraction is reused
within each input preparation pass. The existing five-Agent graph executes the
cheap checks and promotion; no standalone statistical result is substituted for
canonical Quantitative measurements.

At most the existing three-candidate cap is used for training-only construction
planning, in frozen discovery order, before model outcomes. Each proposal is
Agent-owned and reviewed using actual training pairs. Only accepted proposals
receive structured OOD specs. Cheap association checks and OOD feasibility are
separate: a failed construction cannot suppress another usable cheap check.
The existing paired candidate budget, model/seed envelope and sequential gates
remain authoritative; association is not a hard absence veto.

Qualitative uses the task label for balanced groups with the same40-record
inspection budget. The full selected40 may be shown, rather than only24. Labels
are comparison groups, never shortcut measurements. Label groups do not imply
independent acquisition sources. Insufficient class/sample coverage retains the
existing bounded-sampling limitations. Candidate-bound metadata authorization
and existing sample/descriptor identity checks remain active.

`candidate-validation-dispositions.json` distinguishes completed cheap checks,
unavailable/constant measurements, unexecuted cheap checks, deferred construction
and planning budget exhaustion. It does not rewrite canonical SummaryReport
verdicts. An unexecuted feasible check is exposed as such, not claimed completed.

The new entry also opts into `evidence-context-v1` to avoid the large-context
provider failure observed during E3. All raw tool results, predictions, pixel
evidence, metrics and uncertainty remain persisted. Long arrays and reference
lists are summarized only for quantitative/Summarize provider delivery. Actual
model request hashes cover delivered payloads; events record original/delivered
hashes and omitted-array hashes. Legacy clients do not enable this policy.

Windows local invocation (replace the run ID and root for each new experiment):

```powershell
& 'E:\Anaconda_envs\envs\shortcutminer\python.exe' -X utf8 infrastructure/experiments/run_candidate_complete_image_audit.py --input <training-only-public-directory> --training-input <train-val-id-public-directory> --profile configs/experiments/synthetic-corner-patch-shortcutminer-memory-v4.json --runs-root <new-runs-root> --run-id <new-run-id> --task <public-task-description> --isolated-discovery --validation
```

Acceptance tests execute real cheap statistics through the five-Agent graph,
verify four independently bound candidates and ten label groups with no task
training, preserve legacy single-binding output, reject label substitution,
incomplete/duplicate plans and modified source/view hashes, and retain full
canonical evidence during context delivery. These are implementation checks,
not new E1/E2/E3 scientific findings. Historical findings remain tied to their
original frozen sources, configurations and documented limitations.

Accepted executable OOD candidates must include task-model training/evaluation/OOD plan steps in the Executive draft. The opt-in client validates that contract with bounded schema retries. The shared caller-required list contains only association_test; unapproved candidates are not forced to train. Promotion and sequential gates still decide actual execution.
