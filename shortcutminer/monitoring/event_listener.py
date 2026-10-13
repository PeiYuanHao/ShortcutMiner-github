"""Incrementally consume append_event's JSONL and existing model telemetry."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from shortcutminer.protocol.model_runtime import ModelInvocationRecord
from shortcutminer.storage.paths import get_run_dir

from .progress import ProgressEvent, ProgressState, CandidateStatus, PlanStepStatus
from .safety import safe_text, failure_reason


def parse_event(line: str | bytes) -> ProgressEvent:
    """Validate the logging envelope while allowing future event names/details."""
    try:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("event must be an object")
        for key in ("timestamp", "run_id", "event"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"missing or invalid {key}")
        timestamp = datetime.fromisoformat(value["timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must include timezone")
        stage, details = value.get("stage"), value.get("details", {})
        if stage is not None and not isinstance(stage, str):
            raise ValueError("invalid stage")
        if not isinstance(details, dict):
            raise ValueError("invalid details")
        return ProgressEvent(timestamp.astimezone(timezone.utc), value["run_id"], value["event"], stage, details)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("invalid progress event") from exc


class EventListener:
    """Read-only file follower. A fresh instance replays history before tailing.

    Incomplete final lines are held until newline; damaged complete records are
    counted and skipped, so monitoring cannot prevent execution or recovery.
    """

    def __init__(self, runs_root: str | Path, run_id: str):
        self.run_dir = get_run_dir(runs_root, run_id)
        self.state = ProgressState(run_id)
        self._offset = 0
        self._pending = b""

    def _json(self, relative):
        path = self.run_dir / relative
        if not path.resolve().is_relative_to(self.run_dir):
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, UnicodeError):
            return None

    def _checkpoint(self):
        """Read committed LangGraph values without saver.setup or database writes."""
        path = self.run_dir / "checkpoints.sqlite"
        if not path.is_file() or not path.resolve().is_relative_to(self.run_dir):
            return {}
        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1) as connection:
                row = connection.execute(
                    "SELECT checkpoint_id, type, checkpoint FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1",
                    (self.state.run_id,)).fetchone()
                if row is None:
                    return {}
                serializer = JsonPlusSerializer(allowed_msgpack_modules=set())
                snapshot = serializer.loads_typed((row[1], row[2]))
                values = dict(snapshot.get("channel_values", {}))
                # Completed parallel writes may not yet be folded into the next checkpoint.
                for channel, kind, blob in connection.execute(
                    "SELECT channel,type,value FROM writes WHERE thread_id=? AND checkpoint_ns='' AND checkpoint_id=? ORDER BY task_id,idx",
                    (self.state.run_id, row[0])):
                    if channel in {"identify_result", "screening_result", "audit_capabilities", "executive_result",
                                   "quantitative_experiment_plans", "quantitative_result", "qualitative_result", "stage_status"}:
                        value = serializer.loads_typed((kind, blob))
                        if channel == "stage_status":
                            values[channel] = {**values.get(channel, {}), **value}
                        else:
                            values[channel] = value
                return values
        except Exception:
            return {}

    def _artifacts(self):
        state = self.state
        state.artifact_dir = str(self.run_dir)
        snapshot = {**(self._json("workflow_state.json") or {}), **self._checkpoint()}
        if snapshot.get("run_id") not in {None, state.run_id}:
            snapshot = {}
        audit = self._json("audit_request.json")
        if audit:
            state.dataset = safe_text(audit.get("dataset_path"))
            if audit.get("dataset_info") and state.stages["audit_intake"] != "failed":
                state.stages["audit_intake"] = "completed"
        profile = self._json("dataset/profile.json") or snapshot.get("dataset_profile")
        if profile:
            state.dataset = state.dataset or safe_text(profile.get("dataset", {}).get("root_reference"))
            if state.stages["dataset_profiling"] != "failed":
                state.stages["dataset_profiling"] = "completed"
            if profile.get("structure", {}).get("modality") not in {"image", "mixed"}:
                state.stages["visual_discovery"] = "skipped"
        if self._json("identify/visual_discovery.json"):
            state.stages["visual_discovery"] = "completed"
        for stage, status in snapshot.get("stage_status", {}).items():
            stage = {"screening": "candidate_screening", "capabilities": "capability_discovery"}.get(stage, stage)
            if stage in state.stages and state.stages[stage] not in {"running", "failed"}:
                state.stages[stage] = status
        candidates = self._json("identify/output.json") or snapshot.get("identify_result") or {}
        if candidates.get("run_id") not in {None, state.run_id}:
            candidates = {}
        for item in candidates.get("candidates", []):
            identifier = item.get("candidate_id")
            if isinstance(identifier, str):
                state.candidates.setdefault(identifier, CandidateStatus(identifier, safe_text(item.get("cue", {}).get("name"))))
        screening = self._json("screening/result.json") or snapshot.get("screening_result")
        if screening and screening.get("run_id") not in {None, state.run_id}:
            screening = None
        if screening:
            state.stages["candidate_screening"] = "completed"
            for result in screening.get("results", []):
                candidate = state.candidates.get(result.get("candidate_id"))
                if candidate:
                    candidate.decision = result.get("decision") if result.get("decision") in {"promote", "screened_out"} else "pending"
                    candidate.reasons = [failure_reason({"reason": reason}) for reason in result.get("reason", [])
                                         if candidate.decision == "screened_out"]
        capabilities = self._json("capabilities/result.json") or snapshot.get("audit_capabilities")
        promotion = self._json("promotion/result.json") or snapshot.get("promotion_result")
        if promotion and promotion.get("run_id") == state.run_id:
            state.set_promotion(promotion.get("results", []))
            state.stages["stage1_validation"] = "completed"
            if state.stages["evidence_promotion"] != "failed":
                state.stages["evidence_promotion"] = "completed"
        stage2 = snapshot.get("stage2_validation")
        if stage2 and stage2.get("status") in {"completed", "skipped"}:
            state.stages["stage2_validation"] = stage2["status"]
        if capabilities and capabilities.get("run_id") not in {None, state.run_id}:
            capabilities = None
        if capabilities:
            state.capability_available = True
            state.stages["capability_discovery"] = "completed"
        plan = self._json("executive/routing.json") or snapshot.get("executive_result")
        if plan and plan.get("run_id") not in {None, state.run_id}:
            plan = None
        if plan:
            state.set_plan(plan.get("tasks", []))
            if not plan.get("tasks") and screening and screening.get("promoted") == 0:
                state.stages["executive"] = state.agents["executive"] = "skipped"
        for task in list(state.tasks.values()):
            # Canonical task IDs alone may select files; candidate/provider prose cannot.
            import re
            if not re.fullmatch(r"TASK-C[0-9]{3}-(QT|QL)", task.task_id):
                continue
            prefix = f"{task.target}/{task.task_id}"
            experiment_plan = (self._json(prefix + "/experiment_plan.json") or
                               snapshot.get("quantitative_experiment_plans", {}).get(task.task_id))
            if experiment_plan and experiment_plan.get("task_id") == task.task_id:
                for step in experiment_plan.get("steps", []):
                    identifier = step.get("step_id")
                    if isinstance(identifier, str):
                        value = task.steps.setdefault(identifier, PlanStepStatus(identifier, safe_text(step.get("experiment_type")), safe_text(step.get("planning_check"))))
                        if value.status == "pending" and step.get("execution_state") in {"completed", "failed"}:
                            value.status = step["execution_state"]
            result = self._json(prefix + "/result.json")
            if result and result.get("run_id") == state.run_id and result.get("payload", {}).get("task_id") == task.task_id and task.status != "failed":
                task.status = "completed"
        stage_plan = self._json("validation/stage_plan.json") or snapshot.get("validation_stage_plan")
        if stage_plan and stage_plan.get("run_id") == state.run_id:
            for task_id, plan in snapshot.get("quantitative_experiment_plans", {}).items():
                task = state.tasks.get(task_id)
                if task is None:
                    continue
                candidate = state.candidates.get(task.candidate_id)
                for step in plan.get("steps", []):
                    identifier = step["step_id"]
                    value = task.steps.setdefault(identifier, PlanStepStatus(identifier, safe_text(step["experiment_type"]), safe_text(step["planning_check"])))
                    if stage_plan["steps"].get(identifier) == "expensive_validation" and candidate and candidate.promotion in {"terminated_low_signal", "deferred_insufficient_evidence"}:
                        value.status = "skipped"
                        task.reason = "Stage-2 not executed: " + candidate.promotion
        # Read only ToolResult headers/correlation. Never render output, metadata, or error.message.
        for path in (self.run_dir / "tools").glob("*/*.json"):
            result = self._json(path.relative_to(self.run_dir))
            if result and result.get("run_id") == state.run_id:
                details = {key: result.get(key) for key in ("tool_call_id", "candidate_id", "task_id", "plan_step_id", "experiment_type", "duration_ms")}
                details["tool"] = result.get("tool_name")
                name = "TOOL_COMPLETED" if result.get("status") == "success" else "TOOL_FAILED"
                try:
                    stamp = datetime.fromisoformat(result["completed_at"])
                    last_event, current_stage = state.last_event_at, state.current_stage
                    state.apply(ProgressEvent(stamp, state.run_id, name, f"tool:{result.get('agent_name')}", details))
                    state.last_event_at, state.current_stage = last_event, current_stage
                except (ValueError, KeyError, TypeError):
                    pass
        for target in ("quantitative", "qualitative"):
            tasks = [task for task in state.tasks.values() if task.target == target]
            for task in tasks:
                if task.task_id in state._active.get(target, set()):
                    task.status = "running"  # A persisted Stage-1 result cannot finish an active Stage-2 task.
            if tasks:
                status = ("failed" if any(task.status == "failed" for task in tasks) else
                          "completed" if all(task.status == "completed" for task in tasks) else
                          "running" if any(task.status == "running" for task in tasks) else "pending")
                state.stages[target] = status
                state.agents[target] = "waiting" if status == "pending" else status
        if self._json("final_result.json"):
            state.stages["audit_report"] = "completed"
        if state.status == "completed":
            state.current_stage = "completed"

    def poll(self) -> ProgressState:
        path = self.run_dir / "events.jsonl"
        try:
            with path.open("rb") as stream:
                if path.stat().st_size < self._offset:
                    self._offset = 0
                    self._pending = b""
                    self.state = ProgressState(self.state.run_id)
                stream.seek(self._offset)
                data = stream.read()
                self._offset = stream.tell()
        except FileNotFoundError:
            data = b""
        lines = (self._pending + data).split(b"\n")
        self._pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                self.state.apply(parse_event(line))
            except ValueError:
                self.state.ignored_records += 1
        self._artifacts()
        records = []
        for telemetry in (self.run_dir / "model_calls").glob("*.json"):
            if telemetry.name == "telemetry.json" or telemetry.name.startswith("runtime_snapshot-"):
                continue
            try:
                records.append(ModelInvocationRecord.model_validate_json(telemetry.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                # An atomic replacement may race the reader; retry next poll.
                continue
        self.state.update_telemetry(records)
        return self.state
