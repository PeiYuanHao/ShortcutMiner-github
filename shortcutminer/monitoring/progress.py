"""Read-only presentation projection, never an execution state machine."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from .safety import failure_reason, safe_text

AGENTS = ("identify", "executive", "quantitative", "qualitative", "summarize")


class AuditStage(StrEnum):
    audit_intake = "audit_intake"
    dataset_profiling = "dataset_profiling"
    visual_discovery = "visual_discovery"
    identify = "identify"
    candidate_screening = "candidate_screening"
    capability_discovery = "capability_discovery"
    executive = "executive"
    feasibility = "feasibility"
    stage1_validation = "stage1_validation"
    quantitative = "quantitative"
    qualitative = "qualitative"
    evidence_promotion = "evidence_promotion"
    stage2_validation = "stage2_validation"
    evidence_calibration = "evidence_calibration"
    summarize = "summarize"
    audit_report = "audit_report"


STAGE_LABELS = dict(zip(AuditStage, ("Dataset Intake", "Dataset Profiling", "Visual Discovery",
    "Identify Agent", "Candidate Screening", "Capability Discovery", "Executive Agent", "Feasibility",
    "Stage-1 Cheap Validation", "Quantitative Validation", "Qualitative Validation", "Evidence Promotion", "Stage-2 Expensive Validation", "Evidence Calibration", "Summarize", "Audit Report")))
ALIASES = {"screening": "candidate_screening", "capabilities": "capability_discovery"}
WEIGHTS = {"audit_intake": 5, "dataset_profiling": 4, "visual_discovery": 1, "identify": 15,
           "candidate_screening": 5, "capability_discovery": 5, "executive": 10, "feasibility": 0,
           "evidence_calibration": 10, "summarize": 5, "audit_report": 5}


@dataclass(frozen=True)
class ProgressEvent:
    timestamp: datetime
    run_id: str
    event: str
    stage: str | None = None
    details: dict = field(default_factory=dict)


@dataclass
class PlanStepStatus:
    step_id: str
    experiment_type: str
    planning_check: str = ""
    status: str = "pending"
    tool_call_id: str | None = None
    changed_at: datetime | None = None


@dataclass
class TaskStatus:
    task_id: str
    candidate_id: str
    target: str
    requested_checks: list[str] = field(default_factory=list)
    status: str = "pending"
    steps: dict[str, PlanStepStatus] = field(default_factory=dict)
    reason: str | None = None

    @property
    def fraction(self):
        if self.status in {"completed", "skipped"}:
            return 1.0
        return sum(step.status == "completed" for step in self.steps.values()) / len(self.steps) if self.steps else 0.0


@dataclass
class CandidateStatus:
    candidate_id: str
    cue_name: str
    decision: str = "pending"
    reasons: list[str] = field(default_factory=list)
    promotion: str = "pending"
    promotion_reasons: list[str] = field(default_factory=list)


@dataclass
class ToolStatus:
    agent: str
    name: str
    status: str
    duration_ms: float | None = None
    candidate_id: str | None = None
    task_id: str | None = None
    experiment_type: str | None = None
    plan_step_id: str | None = None
    changed_at: datetime | None = None


@dataclass
class ProgressState:
    run_id: str
    dataset: str | None = None
    artifact_dir: str | None = None
    status: str = "waiting"
    current_stage: str = "waiting for events"
    agents: dict = field(default_factory=lambda: dict.fromkeys(AGENTS, "waiting"))
    stages: dict = field(default_factory=lambda: dict.fromkeys(AuditStage, "pending"))
    candidates: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    provider_calls: dict = field(default_factory=lambda: {"text_agent": 0, "semantic_vlm": 0})
    provider_observed: set = field(default_factory=set)
    agent_calls: dict = field(default_factory=dict)
    agent_started: dict = field(default_factory=dict)
    agent_elapsed: dict = field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    vlm_images: int | None = None
    latency_ms: float = 0.0
    telemetry_calls: int = 0
    ignored_records: int = 0
    started_at: datetime | None = None
    last_event_at: datetime | None = None
    ended_at: datetime | None = None
    reason: str | None = None
    planning_complete: bool = False
    capability_available: bool = False
    _active: dict = field(default_factory=dict, repr=False)
    _failed: set = field(default_factory=set, repr=False)
    _provider_event_calls: dict = field(default_factory=dict, repr=False)
    _provider_record_calls: dict = field(default_factory=dict, repr=False)
    _agent_event_calls: dict = field(default_factory=dict, repr=False)
    _agent_record_calls: dict = field(default_factory=dict, repr=False)

    @property
    def completed_agents(self):
        return sum(value == "completed" for value in self.agents.values())

    @property
    def progress(self):
        if self.status == "completed":
            return 1.0
        score = sum(weight for stage, weight in WEIGHTS.items() if self.stages[stage] in {"completed", "skipped"})
        if self.planning_complete:
            score += 35 * sum(task.fraction for task in self.tasks.values()) / len(self.tasks) if self.tasks else 35
        return min(score / 100, 1.0)

    @property
    def current_activity(self):
        active = [tool for tool in self.tools.values() if tool.status == "running"]
        if active:
            return " | ".join(" -> ".join(filter(None, (tool.candidate_id, tool.task_id, tool.name,
                               tool.experiment_type, tool.plan_step_id))) for tool in active)
        return " | ".join(task.task_id for task in self.tasks.values() if task.status == "running") or self.current_stage

    def elapsed_seconds(self, now=None):
        return max(0.0, ((self.ended_at or now or datetime.now(timezone.utc)) - self.started_at).total_seconds()) if self.started_at else 0.0

    def agent_seconds(self, agent, now=None):
        elapsed = self.agent_elapsed.get(agent, 0.0)
        if agent in self.agent_started:
            elapsed += max(0.0, ((self.ended_at or now or datetime.now(timezone.utc)) - self.agent_started[agent]).total_seconds())
        return elapsed

    def interrupt(self):
        if self.status == "running":
            self.status, self.ended_at = "interrupted", self.last_event_at
            self._stop_active("interrupted")

    def _stop_active(self, status):
        end = self.ended_at or self.last_event_at
        if end:
            for agent, start in self.agent_started.items():
                self.agent_elapsed[agent] = self.agent_elapsed.get(agent, 0.0) + max(0.0, (end - start).total_seconds())
        self.agent_started.clear()
        for agent, value in self.agents.items():
            if value == "running":
                self.agents[agent] = status
        for tool in self.tools.values():
            if tool.status == "running":
                tool.status = status
        self._active.clear()

    def set_plan(self, tasks):
        self.planning_complete = True
        for item in tasks:
            identifier = item.get("task_id")
            if isinstance(identifier, str) and item.get("target") in {"quantitative", "qualitative"}:
                task = self.tasks.setdefault(identifier, TaskStatus(identifier, item["candidate_id"], item["target"]))
                task.requested_checks = [safe_text(check) for check in item.get("requested_checks", [])]
        for target in ("quantitative", "qualitative"):
            if not any(task.target == target for task in self.tasks.values()):
                self.stages[target] = self.agents[target] = "skipped"

    def set_promotion(self, results):
        for result in results:
            candidate = self.candidates.get(result.get("candidate_id"))
            if candidate and result.get("decision") in {"promoted", "terminated_low_signal", "deferred_insufficient_evidence"}:
                candidate.promotion = result["decision"]
                candidate.promotion_reasons = [safe_text(reason) for reason in result.get("reasons", [])]

    def apply(self, record):
        if record.run_id != self.run_id:
            return
        self.last_event_at = record.timestamp
        name, stage, details = record.event, ALIASES.get(record.stage, record.stage), record.details
        if name in {"RUN_STARTED", "RUN_RESUMED"}:
            if name == "RUN_RESUMED":
                self._failed.clear()
            self.started_at = self.started_at or record.timestamp
            self.ended_at, self.reason, self.status = None, None, "running"
        elif name in {"RUN_COMPLETED", "RUN_FAILED"}:
            self.status = "completed" if name == "RUN_COMPLETED" else "failed"
            self.ended_at = record.timestamp
            if self.status == "failed":
                self.reason = failure_reason(details)
                for key, value in self.stages.items():
                    if value == "running":
                        self.stages[key] = "failed"
            self._stop_active("interrupted" if self.status == "completed" else "failed")
            if self.status == "completed":
                self.current_stage = "completed"
        if name.startswith("STAGE_") and stage in self.stages:
            self.stages[stage] = {"STAGE_STARTED": "running", "STAGE_COMPLETED": "completed",
                "STAGE_SKIPPED": "skipped", "STAGE_FAILED": "failed"}.get(name, self.stages[stage])
            self.started_at = self.started_at or record.timestamp
            self.current_stage = stage
            if name == "STAGE_FAILED":
                self.reason = failure_reason(details)
            if stage in AGENTS and name != "STAGE_STARTED":
                self.agents[stage] = self.stages[stage]
        elif name.startswith("TASK_") and stage in AGENTS and isinstance(details.get("task_id"), str):
            identifier = details["task_id"]
            task = self.tasks.setdefault(identifier, TaskStatus(identifier, details.get("candidate_id", "N/A"), stage))
            task.status = {"TASK_STARTED": "running", "TASK_COMPLETED": "completed", "TASK_FAILED": "failed"}.get(name, task.status)
            task.requested_checks = [safe_text(check) for check in details.get("requested_checks", task.requested_checks)]
            if name == "TASK_FAILED":
                task.reason = task.reason or failure_reason(details)
        elif stage in AGENTS and name in {f"{stage.upper()}_STARTED", f"{stage.upper()}_COMPLETED", "AGENT_FAILED"}:
            self.current_stage = stage
            active = self._active.setdefault(stage, set())
            key = str(details.get("task_id", stage))
            if name.endswith("_STARTED"):
                if not active:
                    self.agent_started[stage] = record.timestamp
                active.add(key)
                self.agents[stage] = self.stages[stage] = "running"
            else:
                active.discard(key)
                if name == "AGENT_FAILED":
                    self._failed.add(stage)
                if not active and stage in self.agent_started:
                    self.agent_elapsed[stage] = self.agent_elapsed.get(stage, 0.0) + max(0.0, (record.timestamp - self.agent_started.pop(stage)).total_seconds())
                self.agents[stage] = "failed" if stage in self._failed else "running" if active else "completed"
                self.stages[stage] = self.agents[stage]
            task = self.tasks.get(details.get("task_id"))
            if task is not None:
                task.status = "running" if name.endswith("_STARTED") else "failed" if name == "AGENT_FAILED" else "completed"
                if task.status == "failed":
                    task.reason = failure_reason(details)
        elif name == "PROMOTION_COMPLETED":
            self.set_promotion(details.get("results", []))
        elif name in {"CANDIDATE_SCREENING_STARTED", "CANDIDATE_SCREENING_COMPLETED", "AUDIT_CAPABILITIES_BUILT"}:
            key = "capability_discovery" if name == "AUDIT_CAPABILITIES_BUILT" else "candidate_screening"
            self.stages[key] = "running" if name.endswith("STARTED") else "completed"
        elif name == "MODEL_INVOCATION_STARTED":
            kind = details.get("runtime_kind")
            if kind in self.provider_calls:
                self._provider_event_calls[kind] = self._provider_event_calls.get(kind, 0) + 1
                self.provider_calls[kind] = max(self._provider_event_calls[kind], self._provider_record_calls.get(kind, 0))
                self.provider_observed.add(kind)
            if stage in AGENTS:
                self._agent_event_calls[stage] = self._agent_event_calls.get(stage, 0) + 1
                self.agent_calls[stage] = max(self._agent_event_calls[stage], self._agent_record_calls.get(stage, 0))
                self.current_stage = stage
            elif stage == "semantic_visual":
                self.current_stage = "semantic visual inspection"
        elif name in {"TOOL_STARTED", "TOOL_COMPLETED", "TOOL_FAILED", "TOOL_REJECTED"}:
            agent = (stage or "unknown").removeprefix("tool:")
            call_id, tool = details.get("tool_call_id"), details.get("tool")
            if isinstance(call_id, str) and isinstance(tool, str):
                old = self.tools.get((agent, call_id))
                if old and old.changed_at and record.timestamp < old.changed_at:
                    return
                status = {"TOOL_STARTED": "running", "TOOL_COMPLETED": "completed", "TOOL_FAILED": "failed", "TOOL_REJECTED": "rejected"}[name]
                value = ToolStatus(agent, safe_text(tool), status, details.get("duration_ms"))
                value.changed_at = record.timestamp
                for attribute in ("candidate_id", "task_id", "experiment_type", "plan_step_id"):
                    raw = details.get(attribute) or (getattr(old, attribute) if old else None)
                    setattr(value, attribute, safe_text(raw) if isinstance(raw, str) else None)
                self.tools[agent, call_id] = value
                task = self.tasks.get(value.task_id)
                if task is not None and value.plan_step_id:
                    step = task.steps.setdefault(value.plan_step_id, PlanStepStatus(value.plan_step_id, value.experiment_type or "N/A"))
                    if step.changed_at is None or record.timestamp >= step.changed_at:
                        step.status, step.tool_call_id, step.changed_at = status, call_id, record.timestamp
                    if status in {"failed", "rejected"}:
                        task.reason = "backend_execution_failed"
                if agent in AGENTS:
                    self.current_stage = agent

    def update_telemetry(self, records):
        values = {record.invocation_id: record for record in records if record.run_id == self.run_id}
        issued = [record for record in values.values() if record.status != "budget_exceeded"]
        self.telemetry_calls = len(issued)
        for kind in self.provider_calls:
            matching = [record for record in issued if record.runtime_kind == kind]
            if matching:
                self.provider_observed.add(kind)
            self._provider_record_calls[kind] = len(matching)
            self.provider_calls[kind] = max(self._provider_event_calls.get(kind, 0), len(matching))
        for agent in AGENTS:
            matching = [record for record in issued if record.agent_name == agent]
            if matching:
                self._agent_record_calls[agent] = len(matching)
                self.agent_calls[agent] = max(self._agent_event_calls.get(agent, 0), len(matching))
        vlm = [record for record in issued if record.runtime_kind == "semantic_vlm"]
        self.vlm_images = sum(len(record.image_inputs) for record in vlm) if vlm else None
        self.latency_ms = sum(record.latency_ms for record in issued)
        available = bool(issued) and all(record.usage.available for record in issued)
        for attribute in ("prompt_tokens", "completion_tokens", "total_tokens"):
            setattr(self, attribute, sum(getattr(record.usage, attribute) or 0 for record in issued) if available else None)

    @classmethod
    def reconstruct(cls, run_id, records, *, interrupted=False):
        state = cls(run_id)
        for record in records:
            state.apply(record)
        if interrupted:
            state.interrupt()
        return state
