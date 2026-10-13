"""One monitor, with Rich Live on TTY and structured lines on redirected stderr."""
from threading import Event, Thread
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from .event_listener import EventListener
from .progress import ProgressState, STAGE_LABELS
from .safety import safe_text

SYMBOLS = {"pending": "○", "waiting": "○", "running": "▶", "completed": "✓", "failed": "✗", "skipped": "–"}


def value(number):
    return "N/A" if number is None else str(number)


def projection(state):
    """Shared safe strings for both renderers; no raw object dumps."""
    lines = {"header": f"ShortcutMiner progress | ShortcutMiner Audit run={safe_text(state.run_id)} dataset={safe_text(state.dataset)}"}
    for stage, status in state.stages.items():
        lines[f"stage.{stage}"] = f"{stage} {status}"
    for candidate in state.candidates.values():
        reasons = " reason=" + ",".join(candidate.reasons) if candidate.reasons else ""
        lines[f"candidate.{candidate.candidate_id}"] = f"{safe_text(candidate.candidate_id)} {safe_text(candidate.cue_name)} {candidate.decision}{reasons}"
        if candidate.promotion != "pending":
            lines[f"promotion.{candidate.candidate_id}"] = f"{safe_text(candidate.candidate_id)} evidence promotion={candidate.promotion} reason={','.join(candidate.promotion_reasons)}"
    if state.candidates:
        promoted = sum(candidate.decision == "promote" for candidate in state.candidates.values())
        screened = sum(candidate.decision == "screened_out" for candidate in state.candidates.values())
        lines["screening"] = f"screening generated={len(state.candidates)} promoted={promoted} screened_out={screened}" + (" Early termination active" if screened else "")
        evidence_promoted = sum(candidate.promotion == "promoted" for candidate in state.candidates.values())
        terminated = sum(candidate.promotion == "terminated_low_signal" for candidate in state.candidates.values())
        deferred = sum(candidate.promotion == "deferred_insufficient_evidence" for candidate in state.candidates.values())
        lines["promotion"] = f"Generated candidates={len(state.candidates)} Pre-screen promoted={promoted} Evidence-promoted={evidence_promoted} Early terminated before expensive validation={terminated} Deferred={deferred}"
    for task in state.tasks.values():
        reason = f" reason={task.reason}" if task.reason else ""
        lines[f"task.{task.task_id}"] = f"{safe_text(task.task_id)} candidate={safe_text(task.candidate_id)} {task.status} checks={','.join(safe_text(check) for check in task.requested_checks)}{reason}"
        for step in task.steps.values():
            lines[f"step.{task.task_id}.{step.step_id}"] = f"{safe_text(task.task_id)} step={safe_text(step.step_id)} {safe_text(step.experiment_type)} {step.status} tool_call={safe_text(step.tool_call_id)}"
    for agent, status in state.agents.items():
        duration = f"{state.agent_seconds(agent):.1f}s" if agent in state.agent_elapsed or agent in state.agent_started else "N/A"
        lines[f"agent.{agent}"] = f"{agent.capitalize()} {status} calls={value(state.agent_calls.get(agent))} elapsed={duration}"
    for key, tool in state.tools.items():
        lines[f"tool.{key}"] = f"tool={safe_text(tool.name)} {tool.status} task={safe_text(tool.task_id)} step={safe_text(tool.plan_step_id)} duration={value(tool.duration_ms)}ms"
    calls = [value(state.provider_calls[kind]) if kind in state.provider_observed else "N/A" for kind in ("text_agent", "semantic_vlm")]
    latency = f"{state.latency_ms / 1000:.2f}s" if state.telemetry_calls else "N/A"
    lines["runtime"] = (f"Text API calls={calls[0]} VLM calls={calls[1]} Images={value(state.vlm_images)} "
        f"prompt={value(state.prompt_tokens)} completion={value(state.completion_tokens)} total={value(state.total_tokens)} "
        f"latency={latency} tool_calls={len(state.tools)}")
    lines["overall"] = f"Overall ~{state.progress:.0%} (coarse) Elapsed={state.elapsed_seconds():.0f}s status={state.status}"
    lines["activity"] = "Current activity: " + safe_text(state.current_activity, 500)
    if state.reason:
        lines["failure"] = f"Reason: {state.reason} | Artifacts preserved: {safe_text(state.artifact_dir)}"
    return lines


def render_dashboard(state: ProgressState):
    lines = projection(state)
    stages = Table("Pipeline", "Status", expand=True)
    for stage, status in state.stages.items():
        stages.add_row(Text(f"{SYMBOLS.get(status, '○')} {STAGE_LABELS[stage]}"), Text(status))
    agents = Table("Agent", "Status", "Provider calls", "Elapsed", expand=True)
    for agent, status in state.agents.items():
        duration = f"{state.agent_seconds(agent):.1f}s" if agent in state.agent_elapsed or agent in state.agent_started else "N/A"
        agents.add_row(agent.capitalize(), status, value(state.agent_calls.get(agent)), duration)
    candidates = Table("Candidate / Task / PlanStep", "State", expand=True)
    for candidate in state.candidates.values():
        candidates.add_row(Text(f"{safe_text(candidate.candidate_id)} {safe_text(candidate.cue_name)}"), Text(candidate.decision + (": " + ", ".join(candidate.reasons) if candidate.reasons else "")))
        if candidate.promotion != "pending":
            candidates.add_row("  Evidence promotion", Text(candidate.promotion + ": " + ", ".join(candidate.promotion_reasons)))
        for task in state.tasks.values():
            if task.candidate_id == candidate.candidate_id:
                candidates.add_row(Text(f"  {safe_text(task.task_id)} checks={','.join(task.requested_checks)}"), Text(task.status + (": " + task.reason if task.reason else "")))
                for step in task.steps.values():
                    candidates.add_row(Text(f"    {safe_text(step.experiment_type)} [{safe_text(step.step_id)}]"), Text(step.status))
    tools = Table("Tool", "Task / PlanStep", "Status", "Duration", expand=True)
    for tool in state.tools.values():
        tools.add_row(Text(safe_text(tool.name)), Text(f"{safe_text(tool.task_id)} / {safe_text(tool.plan_step_id)}"), Text(tool.status), Text(f"{tool.duration_ms:.0f}ms" if isinstance(tool.duration_ms, (int, float)) else "N/A"))
    content = [Text(lines["header"]), ProgressBar(total=100, completed=state.progress * 100), Text(lines["overall"]),
               stages, agents, candidates, Text(lines.get("screening", "Candidates awaiting persisted screening")),
               Text(lines["runtime"]), tools, Text(lines["activity"])]
    content.append(Text(lines.get("promotion", "Evidence promotion pending")))
    if "failure" in lines:
        content.append(Text(lines["failure"]))
    return Panel(Group(*content), title="ShortcutMiner Audit")


class ProgressMonitor:
    def __init__(self, runs_root, run_id, *, console=None, interval=0.2, dataset=None):
        self.listener = EventListener(runs_root, run_id)
        self.listener.state.dataset = safe_text(dataset) if dataset else None
        self.console = console or Console(stderr=True)
        self.interval = interval
        self._stop = Event()
        self._thread = self._live = None
        self._shown = {}

    def _display(self, state):
        if self._live is not None:
            self._live.update(render_dashboard(state), refresh=True)
        else:
            for key, line in projection(state).items():
                if self._shown.get(key) != line:
                    self.console.print(Text("[ShortcutMiner] " + line), soft_wrap=True)
            self._shown = projection(state)

    def _refresh(self):
        try:
            self._display(self.listener.poll())
        except Exception:
            self._stop.set()

    def _follow(self):
        while not self._stop.wait(self.interval):
            self._refresh()

    def __enter__(self):
        try:
            state = self.listener.poll()
            if self.console.is_terminal:
                self._live = Live(render_dashboard(state), console=self.console, auto_refresh=False)
                self._live.start(refresh=True)
            else:
                self._display(state)
            self._thread = Thread(target=self._follow, name="shortcutminer-progress", daemon=True)
            self._thread.start()
        except Exception:
            self._stop.set()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.listener.poll()
            self.listener.state.interrupt()
            self._display(self.listener.state)
        except Exception:
            pass
        finally:
            if self._live is not None:
                try:
                    self._live.stop()
                except Exception:
                    pass
        return False
