# Identify Agent Memory Policy

## 1. Purpose

Memory must support stable operation without contaminating independent experiments.

The Identify Agent may retain general operational knowledge that remains valid across runs.

The Identify Agent must not allow run-specific shortcut findings to influence later independent runs.

Formal experiments require strict run independence.

---

## 2. Allowed Persistent Memory

Persistent memory may contain stable operational information such as:

* the permanent role of the Identify Agent;
* the shortcut cue taxonomy;
* protocol conventions;
* tool-use rules;
* candidate construction rules;
* evidence discipline;
* failure-handling principles;
* stable dataset-access conventions explicitly defined by the framework.

This information describes **how to perform discovery**.

It must not describe **what shortcut was found in a particular experiment**.

---

## 3. Prohibited Persistent Memory

Do not persist run-specific findings such as:

* discovered shortcut candidates;
* candidate priorities;
* dataset-specific observations;
* validation outcomes;
* candidate rankings;
* inspected sample conclusions;
* model-specific behavior;
* seed-specific behavior;
* previous `CandidateReport` contents;
* downstream audit conclusions.

Examples of prohibited persistent memory include:

```text id="76h9m0"
Waterbirds has a background shortcut.
```

```text id="6iw8b3"
Candidate C001 was high priority in seed 42.
```

```text id="bjzbii"
Hospital identity was validated as a shortcut in the previous run.
```

These findings must not become assumptions in a new independent run.

---

## 4. Run-Local Context

Information required during one run may be retained temporarily within that run.

Examples include:

* current `run_id`;
* current dataset specification;
* current task definition;
* selected sample identifiers;
* tool results from the current run;
* temporary observations;
* current candidate drafts;
* current limitations.

Run-local context may be used until the current Identify execution is complete.

It must not be automatically transferred into another independent run.

---

## 5. Cross-Run Independence

For every formal run, behave as if shortcut discovery is being performed independently.

Do not use previous findings to:

* preselect candidates;
* increase candidate priority;
* suppress alternative hypotheses;
* skip inspection steps;
* assume known dataset behavior;
* infer evidence that was not observed in the current run.

A candidate must be supported by evidence from the current run.

---

## 6. Same Dataset, New Run

The same dataset may appear across:

* different seeds;
* different LLM backends;
* different repetitions;
* different experiment configurations.

Each such run must remain independent unless the experiment protocol explicitly defines shared information.

For example:

```text id="3yw2jl"
waterbirds__model-a__seed42__r01
```

and

```text id="uou2i2"
waterbirds__model-b__seed3407__r02
```

must not share previous shortcut conclusions by default.

The fact that the dataset name is the same does not authorize reuse of earlier findings.

---

## 7. Prior Knowledge Provided in the Request

If the current request explicitly provides prior clues, known metadata descriptions, or task context, those may be used.

Such information is treated as **current-run input**, not as hidden memory.

Explicitly provided clues must still be treated according to their stated epistemic status.

A prior clue is not automatically a validated fact.

---

## 8. Framework Knowledge vs. Experimental Evidence

Always distinguish between:

### Framework Knowledge

Stable information about how ShortcutMiner operates.

Examples:

```text id="617aff"
Candidate status must be unverified.
```

```text id="r3zf4r"
Identify does not perform final validation.
```

This may persist.

### Experimental Evidence

Information obtained from a particular dataset run.

Examples:

```text id="wz6jip"
Class 0 samples appeared brighter in this inspection.
```

```text id="jc4kwa"
The current metadata contains a hospital_id field.
```

This must remain run-local unless explicitly recorded as part of an experiment artifact.

---

## 9. Artifact Storage Is Not Agent Memory

Experiment outputs may be stored by ShortcutMiner under the run directory.

Examples include:

```text id="ihwfv9"
runs/<run_id>/identify/input.json
runs/<run_id>/identify/output.json
```

Stored artifacts exist for:

* reproducibility;
* traceability;
* debugging;
* later evaluation.

They must not be treated as hidden prior knowledge for a new Identify run.

Reading previous run artifacts should occur only when explicitly requested by the current experiment protocol.

---

## 10. Memory and Reproducibility

The goal of memory restrictions is to make experimental differences attributable to controlled variables such as:

* model;
* seed;
* repeat;
* dataset;
* configuration.

They should not be caused by uncontrolled accumulated memory.

A formal run should therefore be reproducible from:

```text id="cgk9ll"
configuration
+
current dataset state
+
current run context
+
available tools
+
current model behavior
```

rather than:

```text id="k981qg"
hidden conclusions from previous experiments
```

---

## 11. Memory Failure Rule

If previous-run information appears in memory unexpectedly:

1. do not treat it as current evidence;
2. do not use it to construct or rank candidates;
3. derive conclusions again from current-run evidence;
4. preserve the independence of the current run.

When uncertain whether information belongs to the current run, prefer not to use it.

---

## 12. Permanent Rule

> **Remember how to discover shortcuts, not which shortcuts were discovered before.**
