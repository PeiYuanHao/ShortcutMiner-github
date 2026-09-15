# Executive Agent Invocation Checklist

## Purpose

HEARTBEAT.md defines checks within each invocation, not a background service.
There is no polling, periodic wake-up, task-status loop or Agent-controlled retry.

## Intake

1. Read the single ExecutiveRequest under request.
2. Check run_id, protocol_version, candidate_report and routing_context alignment.
3. Preserve every upstream Candidate exactly; treat each as unverified.
4. Read targets, checks, outputs, sample/model capabilities and constraints from
   the request only. A desired capability is not an available capability.

## Planning

1. Assign one route per candidate: quantitative, qualitative, both or unresolved.
2. State a falsifiable objective and why each selected specialist is needed.
3. Create only complete AnalysisTask objects supported by the capability snapshot.
4. Respect max_tasks, max_samples_per_task, priority and runtime_constraints.
5. Use TASK-Cnnn-QT / TASK-Cnnn-QL IDs; dependencies must be empty in this workflow.
6. Preserve unresolved candidates with no tasks and an explicit limitation or
   missing requirement. Do not invent checks to force an executable plan.

## Output

1. Return one RoutingPlan to orchestrator, correlated to request.message_id.
2. Check ordered candidate coverage, candidate integrity and task correlations.
3. Preserve context limitations and authorized artifact references.
4. Do not claim dispatch, experiment progress, results, retries or final synthesis.
5. The executor validates and persists the plan and events. Finish this invocation;
   do not wait for either specialist or initiate a Summarize handoff.

START → Identify → Executive → (Quantitative ∥ Qualitative) → Summarize → END. Quantitative and Qualitative are parallel branches; Summarize runs after their join. Runner and Orchestrator are infrastructure, not Agents.

## Failure Discipline

An unplannable candidate uses route/status unresolved in RoutingDecision.
Do not emit custom blocked, needs_clarification or heartbeat messages.
Malformed input or invalid output is an execution failure handled by the control
layer; never fabricate a valid-looking plan from missing identifiers.

Before returning a model-reliance plan, verify that the Candidate itself resolves the intended cue/region, every new check declares its required model capabilities, and no training task survives `no_retraining`. Treat unavailable intervention inference or segmentation as unresolved scope, not contradiction.

## Final Structured-Output Rule

Return exactly one valid JSON object matching `RoutingPlan`. Do not return
Markdown, use code fences, or include prose before or after the JSON object. The
first non-whitespace character must be `{` and the last non-whitespace character
must be `}`.
