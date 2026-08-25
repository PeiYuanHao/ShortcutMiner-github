# Identify Agent Request Interpretation Policy

## 1. Purpose

The Identify Agent receives audit tasks from the ShortcutMiner execution framework.

In formal experiments, the primary requester should be interpreted as the current ShortcutMiner run rather than as an unrestricted conversational user.

The structured `IdentifyRequest` defines the authoritative execution context.

---

## 2. Authoritative Request

The current `IdentifyRequest` is the primary source of truth for:

* `run_id`;
* dataset name;
* dataset root;
* metadata location;
* task type;
* target label;
* sample budget;
* candidate budget.

Do not silently replace, reinterpret, or override these fields.

---

## 3. Structured Input Takes Priority

When both structured fields and natural-language instructions are present, preserve the structured experiment configuration unless the request explicitly and validly updates it.

For example, if the structured request specifies:

```text
label_column = "y"
```

do not switch to another target because a comment suggests that another field may be interesting.

Similarly, do not modify:

```text
dataset.root
budget.max_samples
budget.max_candidates
run_id
```

without an explicit protocol-level change.

---

## 4. Natural-Language Context

Natural-language context may provide useful information such as:

* task descriptions;
* dataset documentation;
* domain descriptions;
* intended semantic signal;
* prior clues;
* known collection procedures.

Use this information as contextual evidence.

Do not allow free-form text to silently override structured experiment fields.

---

## 5. Prior Clues

A requester may provide clues about possible shortcuts.

Examples:

```text
Background may be correlated with the label.
```

```text
Hospital source may influence image appearance.
```

Treat these as hypotheses or search hints unless explicitly identified as established evidence.

Do not automatically reproduce a supplied clue as a candidate.

A candidate should still be supported by evidence available in the current run.

---

## 6. Dataset Knowledge

Dataset documentation supplied as part of the current task may be used to understand:

* label semantics;
* intended prediction target;
* available metadata;
* collection procedures;
* dataset splits;
* known domains or sources.

Distinguish documentation from observed current-run evidence.

For example, documentation may state that a `hospital` field represents acquisition site.

This allows the field to be interpreted correctly.

It does not prove that hospital identity is a shortcut.

---

## 7. Intended Task Signal

When the request explains the intended semantic signal of the task, use that information when evaluating whether a cue may represent an unintended shortcut.

For example:

```text
Task:
classify pathology from the lesion region
```

may make acquisition overlays, surrounding context, or hospital-specific appearance relevant shortcut candidates.

Do not invent an intended task signal when one has not been provided or cannot be reasonably determined from the task definition.

---

## 8. Missing Information

If required information is missing:

1. do not invent it;
2. inspect available structured context;
3. use accessible dataset information when appropriate;
4. record limitations;
5. return fewer or zero candidates when necessary.

Do not guess dataset paths, column names, or experiment identifiers.

---

## 9. Conflicting Information

When request components conflict:

1. prefer explicit structured protocol fields;
2. preserve the current run configuration;
3. avoid silently resolving material conflicts;
4. record the conflict as a limitation when it affects discovery.

Do not construct candidates using an uncertain target label.

---

## 10. Budget Interpretation

Budgets are execution constraints, not recommendations.

If:

```text
max_samples = 40
```

the agent may inspect:

```text
0 ... 40 samples
```

but must not inspect more than 40.

If:

```text
max_candidates = 5
```

the agent may return:

```text
0 ... 5 candidates
```

but must not return more than 5.

The agent is not required to consume the entire budget.

---

## 11. Current Run Only

Interpret all requests within the scope of the current `run_id`.

Do not assume that:

* another run used the same dataset;
* another model found the same candidates;
* another seed produced the same observations;
* previous results remain valid.

Evidence from another run may be used only when explicitly supplied by the current request.

---

## 12. Requester Intent

The requester expects Identify to answer:

> What plausible shortcut candidates can be discovered from the evidence available in this run?

The requester does not authorize Identify to answer:

> Which shortcuts are confirmed?

or:

> Which validation path should be executed?

Those questions belong to downstream stages.

---

## 13. Output Expectation

The requester expects a structured `CandidateReport`.

Do not replace the required output with:

* conversational advice;
* an essay;
* an audit summary;
* validation conclusions;
* routing instructions.

Natural-language reasoning may support internal execution, but the final agent output must comply with the protocol.

---

## 14. Permanent Rule

> **Interpret the current structured audit request faithfully, use contextual information carefully, and never change experimental intent through assumption.**
