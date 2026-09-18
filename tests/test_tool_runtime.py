"""Offline integration, permission, safety, and persistence tests for tools."""

import json
from copy import deepcopy
from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from PIL import Image
from pydantic import Field

from shortcutminer.experiment.tool_executor import MockExperimentExecutor
from shortcutminer.protocol.identify import CandidateReport, StrictProtocolModel
from shortcutminer.protocol.quantitative import QuantitativeResult
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.runner import run_workflow
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.tooling.base import StructuredTool
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.errors import (
    ToolBudgetExceeded,
    ToolInputValidationError,
    ToolNotFound,
    ToolPermissionError,
)
from shortcutminer.tooling.protocol import ToolCall, ToolResult
from shortcutminer.tooling.registry import ToolRegistry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime
from shortcutminer.workflow.checkpoint import (
    checkpoint_config,
    checkpoint_path,
    open_sqlite_checkpointer,
)
from shortcutminer.workflow.graph import build_workflow
from test_pipeline_mock import _initial_state


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "images").mkdir()
    Image.new("RGB", (8, 8), (32, 64, 96)).save(root / "images" / "one.png")
    (root / "metadata.csv").write_text(
        "id,label,group,path\n1,0,A,images/one.png\n2,1,B,images/one.png\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def registry():
    return build_default_tool_registry(
        experiment_executor=MockExperimentExecutor(metrics={"score": 0.25})
    )


def _call(
    *,
    tool_name: str,
    agent_name: str = "identify",
    run_id: str = "tool_test_run",
    call_id: str = "tool-call-001",
    arguments: dict | None = None,
    candidate_id: str | None = None,
    task_id: str | None = None,
) -> ToolCall:
    return ToolCall(
        tool_call_id=call_id,
        tool_name=tool_name,
        agent_name=agent_name,
        run_id=run_id,
        candidate_id=candidate_id,
        task_id=task_id,
        arguments=arguments or {},
    )


def test_registry_registration_query_and_provider_neutral_schemas(registry):
    identify_names = {spec.name for spec in registry.specs_for("identify")}
    executive_names = {spec.name for spec in registry.specs_for("executive")}

    assert {"inspect_dataset", "inspect_metadata", "select_samples"} <= identify_names
    assert "run_experiment" not in executive_names
    assert registry.resolve("inspect_dataset").name == "inspect_dataset"
    assert registry.specs_for("identify")[0].input_schema
    with pytest.raises(ValueError, match="already registered"):
        tool = registry.resolve("inspect_dataset")
        registry.register(tool, agents={"identify"})


def test_permission_matrix_is_least_privilege(registry):
    matrix = registry.permission_matrix()

    assert matrix == {
        "get_column_distribution": {
            "identify": True,
            "executive": False,
            "quantitative": True,
            "qualitative": True,
            "summarize": False,
        },
        "inspect_dataset": {
            "identify": True,
            "executive": False,
            "quantitative": True,
            "qualitative": True,
            "summarize": False,
        },
        "inspect_metadata": {
            "identify": True,
            "executive": False,
            "quantitative": True,
            "qualitative": True,
            "summarize": False,
        },
        "inspect_sample_metadata": {
            "identify": True,
            "executive": False,
            "quantitative": True,
            "qualitative": True,
            "summarize": False,
        },
        "inspect_sample": {
            "identify": False,
            "executive": False,
            "quantitative": False,
            "qualitative": True,
            "summarize": False,
        },
        "inspect_tool_capabilities": {
            "identify": True,
            "executive": True,
            "quantitative": True,
            "qualitative": True,
            "summarize": True,
        },
        "read_artifact": {
            "identify": True,
            "executive": True,
            "quantitative": True,
            "qualitative": True,
            "summarize": True,
        },
        "read_run_trace": {
            "identify": False,
            "executive": True,
            "quantitative": True,
            "qualitative": True,
            "summarize": True,
        },
        "run_experiment": {
            "identify": False,
            "executive": False,
            "quantitative": True,
            "qualitative": False,
            "summarize": False,
        },
        "select_samples": {
            "identify": True,
            "executive": False,
            "quantitative": True,
            "qualitative": True,
            "summarize": False,
        },
    }


def test_unknown_and_unauthorized_tools_are_rejected(registry, tmp_path: Path):
    runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="executive",
        runs_root=tmp_path,
    )
    with pytest.raises(ToolNotFound, match="Unknown tool"):
        runtime.execute(_call(tool_name="unknown_tool", agent_name="executive"))
    with pytest.raises(ToolPermissionError, match="cannot call"):
        runtime.execute(
            _call(
                tool_name="run_experiment",
                agent_name="executive",
                call_id="tool-call-002",
                candidate_id="C001",
                task_id="TASK-C001-QT",
                arguments={
                    "candidate_id": "C001",
                    "task_id": "TASK-C001-QT",
                    "experiment_type": "statistical_check",
                },
            )
        )


def test_dataset_metadata_and_sample_tools_wrap_existing_implementations(
    registry, tmp_path: Path
):
    dataset = _dataset(tmp_path / "dataset")
    runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="identify",
        dataset_root=dataset,
        budget=ToolBudget(max_tool_calls=5, max_selected_samples=2),
    )
    inspected = runtime.execute(_call(tool_name="inspect_dataset"))
    profile = runtime.execute(
        _call(
            tool_name="inspect_metadata",
            call_id="tool-call-002",
            arguments={"metadata_file": "metadata.csv", "max_rows": 10},
        )
    )
    distribution = runtime.execute(
        _call(
            tool_name="get_column_distribution",
            call_id="tool-call-004",
            arguments={"metadata_file": "metadata.csv", "column": "label"},
        )
    )
    selection_call = _call(
        tool_name="select_samples",
        call_id="tool-call-003",
        arguments={
            "metadata_file": "metadata.csv",
            "max_samples": 2,
            "seed": 7,
            "strategy": "class_balanced",
            "label_column": "label",
        },
    )
    selected = runtime.execute(selection_call)
    repeat = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="identify",
        dataset_root=dataset,
        budget=ToolBudget(max_selected_samples=2),
    ).execute(selection_call.model_copy(update={"tool_call_id": "repeat-call"}))

    assert inspected.status == "success"
    assert inspected.output["num_files"] == 2
    assert profile.output["num_columns"] == 4
    assert distribution.output["values"] == [
        {"value": 0, "count": 1},
        {"value": 1, "count": 1},
    ]
    assert selected.output == repeat.output


@pytest.mark.parametrize(
    "artifact",
    ["../secret.txt", "/etc/passwd", r"C:\\Windows\\System32\\drivers\\etc\\hosts"],
)
def test_artifact_reader_rejects_traversal_and_system_paths(
    registry, tmp_path: Path, artifact: str
):
    runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="summarize",
        runs_root=tmp_path,
        approved_artifacts={artifact},
    )
    with pytest.raises(ToolInputValidationError):
        runtime.execute(
            _call(
                tool_name="read_artifact",
                agent_name="summarize",
                arguments={"artifact": artifact},
            )
        )


def test_artifact_and_sample_readers_are_bounded_and_allowlisted(
    registry, tmp_path: Path
):
    dataset = _dataset(tmp_path / "dataset")
    run_dir = tmp_path / "runs" / "tool_test_run"
    (run_dir / "approved").mkdir(parents=True)
    (run_dir / "approved" / "result.json").write_text(
        '{"value": 3}', encoding="utf-8"
    )
    artifact_runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="summarize",
        runs_root=tmp_path / "runs",
        approved_artifacts={"approved/result.json"},
    )
    result = artifact_runtime.execute(
        _call(
            tool_name="read_artifact",
            agent_name="summarize",
            arguments={"artifact": "approved/result.json"},
        )
    )
    denied = artifact_runtime.execute(
        _call(
            tool_name="read_artifact",
            agent_name="summarize",
            call_id="tool-call-002",
            arguments={"artifact": "unapproved.json"},
        )
    )
    sample_runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="qualitative",
        dataset_root=dataset,
        approved_samples={"images/one.png"},
    )
    sample = sample_runtime.execute(
        _call(
            tool_name="inspect_sample_metadata",
            agent_name="qualitative",
            arguments={
                "sample_id": "S1",
                "sample_reference": "images/one.png",
            },
        )
    )

    assert result.output["content"] == {"value": 3}
    assert denied.status == "failed"
    assert denied.error.error_type == "ToolExecutionError"
    assert sample.output["sample_available"] is True
    assert sample.output["media_type"] == "image"


def test_inspect_sample_is_qualitative_only_bounded_and_allowlisted(
    registry, tmp_path: Path
):
    dataset = _dataset(tmp_path / "dataset")
    runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="qualitative",
        dataset_root=dataset,
        approved_samples={"images/one.png"},
    )
    result = runtime.execute(
        _call(
            tool_name="inspect_sample",
            agent_name="qualitative",
            arguments={
                "sample_id": "S1",
                "sample_reference": "images/one.png",
                "requested_descriptors": ["dimensions", "rgb_stats"],
                "max_pixels": 64,
            },
        )
    )
    denied = runtime.execute(
        _call(
            tool_name="inspect_sample",
            agent_name="qualitative",
            call_id="tool-call-002",
            arguments={
                "sample_id": "S2",
                "sample_reference": "metadata.csv",
            },
        )
    )

    assert result.status == "success"
    assert set(result.output["visual_descriptors"]) == {"dimensions", "rgb_stats"}
    assert result.output["visual_descriptors"]["dimensions"]["width"] == 8
    assert denied.status == "failed"
    assert "approved reference set" in denied.error.message
    with pytest.raises(ToolInputValidationError, match="requested_descriptors"):
        runtime.execute(
            _call(
                tool_name="inspect_sample",
                agent_name="qualitative",
                call_id="tool-call-003",
                arguments={
                    "sample_id": "S1",
                    "sample_reference": "images/one.png",
                    "requested_descriptors": ["shortcut_score"],
                },
            )
        )


def test_experiment_inconclusive_is_a_successful_tool_execution(
    tmp_path: Path,
):
    backend = MockExperimentExecutor(outcome="inconclusive", metrics={})
    registry = build_default_tool_registry(experiment_executor=backend)
    runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="quantitative",
        runs_root=tmp_path,
    )
    result = runtime.execute(
        _call(
            tool_name="run_experiment",
            agent_name="quantitative",
            candidate_id="C001",
            task_id="TASK-C001-QT",
            arguments={
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "statistical_check",
            },
        )
    )

    assert result.status == "success"
    assert result.output["outcome"] == "inconclusive"
    assert len(backend.calls) == 1


class _TestInput(StrictProtocolModel):
    delay: float = Field(default=0, ge=0)


class _TestOutput(StrictProtocolModel):
    value: int


def _runtime_with_test_tool(tmp_path: Path, handler, timeout: float) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(
        StructuredTool(
            "test_failure",
            "Test-only structured failure tool.",
            _TestInput,
            _TestOutput,
            handler,
        ),
        agents={"identify"},
    )
    return ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="identify",
        runs_root=tmp_path,
        budget=ToolBudget(max_tool_calls=1, timeout_seconds=timeout),
    )


def test_tool_exception_and_timeout_return_structured_failures(tmp_path: Path):
    def fails(_arguments, _context):
        raise RuntimeError("planned tool failure")

    failed = _runtime_with_test_tool(tmp_path / "failed", fails, 1).execute(
        _call(tool_name="test_failure")
    )

    def waits(arguments, _context):
        sleep(arguments.delay)
        return _TestOutput(value=1)

    timed_out = _runtime_with_test_tool(tmp_path / "timeout", waits, 0.01).execute(
        _call(tool_name="test_failure", arguments={"delay": 0.1})
    )

    assert failed.error.error_type == "ToolExecutionError"
    assert timed_out.error.error_type == "ToolTimeoutError"


def test_invalid_tool_output_is_structured_and_budget_is_bounded(tmp_path: Path):
    def invalid_output(_arguments, _context):
        return {"value": "not-an-integer"}

    runtime = _runtime_with_test_tool(tmp_path, invalid_output, 1)
    result = runtime.execute(_call(tool_name="test_failure"))
    assert result.status == "failed"
    assert result.error.error_type == "ToolExecutionError"

    with pytest.raises(ToolBudgetExceeded, match="Duplicate tool_call_id"):
        runtime.execute(_call(tool_name="test_failure"))

    with pytest.raises(ToolBudgetExceeded, match="budget is exhausted"):
        runtime.execute(
            _call(tool_name="test_failure", call_id="tool-call-002")
        )


def test_tool_call_result_trace_and_serialization(registry, tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    runtime = ToolRuntime(
        registry,
        run_id="tool_test_run",
        agent_name="identify",
        dataset_root=dataset,
        runs_root=tmp_path / "runs",
    )
    call = _call(tool_name="inspect_dataset")
    result = runtime.execute(call)
    restored_call = ToolCall.model_validate_json(call.model_dump_json())
    restored_result = ToolResult.model_validate_json(result.model_dump_json())
    result_path = tmp_path / "runs" / "tool_test_run" / result.artifacts[0]
    events = [
        json.loads(line)
        for line in (
            tmp_path / "runs" / "tool_test_run" / "events.jsonl"
        ).read_text("utf-8").splitlines()
    ]

    assert restored_call == call
    assert restored_result == result
    assert ToolResult.model_validate_json(result_path.read_text("utf-8")) == result
    assert [event["event"] for event in events] == [
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]


class ToolCallingDemoClient(DemoAgentClient):
    """Request one permitted tool per Agent before returning real mock schemas."""

    def __init__(self) -> None:
        super().__init__()
        self._tool_lock = Lock()

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        with self._tool_lock:
            self.calls.append(
                {
                    "agent_name": agent_name,
                    "payload": deepcopy(payload),
                    "system_prompt": system_prompt,
                }
            )
        runtime = payload.get("tool_runtime")
        if runtime is not None and not runtime["results"]:
            request = payload["request"]
            if agent_name == "identify":
                return {
                    "tool_calls": [
                        {
                            "name": "inspect_dataset",
                            "arguments": {},
                        },
                        {
                            "name": "inspect_metadata",
                            "arguments": {
                                "metadata_file": request["dataset"][
                                    "metadata_file"
                                ]
                            },
                        },
                    ]
                }
            elif agent_name == "executive":
                call = {
                    "name": "inspect_tool_capabilities",
                    "arguments": {},
                }
            elif agent_name == "quantitative":
                call = {
                    "name": "run_experiment",
                    "arguments": {
                        "experiment_type": "statistical_check",
                        "parameters": {"cue_field": "site"},
                    },
                }
            elif agent_name == "qualitative":
                call = {
                    "name": "inspect_sample_metadata",
                    "arguments": {
                        "sample_id": "S1",
                        "sample_reference": "images/one.png",
                    },
                }
            else:
                call = {"name": "read_run_trace", "arguments": {}}
            return {"tool_calls": [call]}

        output = getattr(self, f"_{agent_name}")(payload)
        return output


def test_five_agent_tool_loop_permissions_checkpoint_and_schema_regression(
    tmp_path: Path,
):
    state = _initial_state(tmp_path)
    state["run_id"] = "tool_workflow_run"
    registry = build_default_tool_registry(
        experiment_executor=MockExperimentExecutor(metrics={"association": 0.75})
    )
    dataset_root = state["dataset_info"]["root"]
    images = Path(dataset_root) / "images"
    images.mkdir()
    (images / "one.png").write_bytes(b"\x89PNG\r\n\x1a\nmock")
    runs_root = state["identify_config"]["runs_root"]
    runtimes = {
        agent: ToolRuntime(
            registry,
            run_id=state["run_id"],
            agent_name=agent,
            dataset_root=dataset_root,
            runs_root=runs_root,
            approved_samples=(
                {"images/one.png"} if agent == "qualitative" else set()
            ),
        )
        for agent in ("identify", "executive", "quantitative", "qualitative", "summarize")
    }
    client = ToolCallingDemoClient()
    result = run_workflow(
        state,
        agent_client=client,
        tool_runtimes=runtimes,
    )

    identify = CandidateReport.model_validate(result["identify_result"])
    quantitative = QuantitativeResult.model_validate(
        result["quantitative_result"]["results"][0]
    )
    summary = SummaryReport.model_validate(result["final_result"])
    experiment_results = runtimes["quantitative"].history
    assert identify.candidates[0].candidate_id == "C001"
    assert quantitative.payload.task_id == "TASK-C001-QT"
    assert quantitative.payload.candidate_id == "C001"
    assert quantitative.payload.artifact_references == [
        artifact for item in experiment_results for artifact in item.artifacts
    ]
    assert all(item.status == "success" for item in experiment_results)
    assert summary.payload.findings[0].candidate_id == "C001"
    assert all(runtime.history for runtime in runtimes.values())
    assert [result.tool_name for result in runtimes["identify"].history] == [
        "inspect_dataset",
        "inspect_metadata",
    ]
    assert all(
        item.tool_call_id.startswith("TC-")
        for runtime in runtimes.values()
        for item in runtime.history
    )
    assert runtimes["identify"].history[0].candidate_id is None
    assert runtimes["executive"].history[0].task_id is None
    assert runtimes["quantitative"].history[0].task_id == "TASK-C001-QT"
    assert runtimes["qualitative"].history[0].task_id == "TASK-C001-QL"
    assert runtimes["summarize"].history[0].candidate_id is None

    database = checkpoint_path(runs_root, state["run_id"])
    with open_sqlite_checkpointer(database) as saver:
        workflow = build_workflow(checkpointer=saver, agent_client=DemoAgentClient())
        restored = workflow.get_state(checkpoint_config(state["run_id"])).values
    restored_quantitative = QuantitativeResult.model_validate(
        restored["quantitative_result"]["results"][0]
    )
    assert restored_quantitative.payload.artifact_references == [
        artifact for item in experiment_results for artifact in item.artifacts
    ]
