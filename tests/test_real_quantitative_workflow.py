from pathlib import Path

from shortcutminer.runner import resume_run, run_workflow
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.tooling.runtime import ToolRuntime
from test_pipeline_mock import _initial_state


class RealQuantitativeClient(DemoAgentClient):
    """Use mocks for reasoning while Quantitative consumes real tool output."""

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        self.calls.append(
            {
                "agent_name": agent_name,
                "payload": payload,
                "system_prompt": system_prompt,
            }
        )
        runtime = payload.get("tool_runtime")
        if agent_name == "quantitative" and runtime and not runtime["results"]:
            request = payload["request"]
            return {
                "tool_calls": [
                    {
                        "name": "run_experiment",
                        "arguments": {
                            "experiment_type": "statistical_check",
                            "parameters": {
                                "cue_field": "site",
                                "n_permutations": 10,
                            },
                        },
                    }
                ]
            }
        output = getattr(self, f"_{agent_name}")(payload)
        if agent_name == "quantitative" and runtime:
            tool_result = runtime["results"][0]
            experiment = tool_result["output"]
            assert experiment["metadata"]["backend"] == "real_cpu_v1"
            output["evidence_summary"] = (
                "The real association output was interpreted without converting "
                "a statistical threshold into a final shortcut judgment."
            )
        return output


def test_full_workflow_uses_real_quantitative_backend_and_resume_reuses_result(
    tmp_path: Path,
):
    state = _initial_state(tmp_path)
    state["run_id"] = "real-quantitative-workflow"
    registry = build_default_tool_registry(
        experiment_executor=RealExperimentExecutor()
    )
    runtime = ToolRuntime(
        registry,
        run_id=state["run_id"],
        agent_name="quantitative",
        dataset_root=state["dataset_info"]["root"],
        runs_root=state["identify_config"]["runs_root"],
    )
    client = RealQuantitativeClient()

    result = run_workflow(
        state,
        agent_client=client,
        tool_runtimes={"quantitative": runtime},
    )

    quantitative = result["quantitative_result"]["results"][0]["payload"]
    assert "cramers_v" in quantitative["metrics"].values()
    assert any(
        key.endswith("effect_size") and value > 0.9
        for key, value in quantitative["metrics"].items()
    )
    experiment_reference = next(
        reference
        for reference in quantitative["artifact_references"]
        if reference.startswith("experiments/")
    )
    artifact = (
        Path(state["identify_config"]["runs_root"])
        / state["run_id"]
        / experiment_reference
    )
    assert artifact.is_file()
    original_mtime = artifact.stat().st_mtime_ns
    assert result["final_result"]["message_type"] == "summary_report"

    resumed = resume_run(
        state["run_id"],
        runs_root=state["identify_config"]["runs_root"],
        agent_client=client,
        tool_runtimes={"quantitative": runtime},
    )
    assert resumed.status == "COMPLETED"
    assert artifact.stat().st_mtime_ns == original_mtime
    assert len(runtime.history) == 1
