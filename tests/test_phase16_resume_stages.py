"""Checkpoint interruptions at the production graph boundaries."""

import json
from collections import Counter
from pathlib import Path

import pytest

import shortcutminer.workflow.graph as graph_module
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import resume_run, run
from shortcutminer.runtime.demo import build_demo_request
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime
from test_phase16_three_candidate_chain import FakeTransport


def _agent_counts(transport):
    return Counter(
        json.loads(call["messages"][-1]["content"])["request"]["target"]
        for call in transport.calls
    )


@pytest.mark.parametrize("interrupted_node", [
    "executive", "feasibility", "quantitative", "qualitative",
    "evidence_calibration", "summarize",
])
def test_resume_does_not_reexecute_checkpointed_stages(
    tmp_path: Path, monkeypatch, interrupted_node: str,
):
    runs_root = tmp_path / "runs"
    run_id = f"resume-{interrupted_node.replace('_', '-')}"
    base = build_demo_request(runs_root, run_id)
    dataset_root = Path(base.dataset_info.root)
    (dataset_root / "metadata.csv").write_text(
        "id,label,site,device,batch\n"
        "1,0,A,X,P\n2,0,A,X,P\n3,1,B,Y,Q\n4,1,B,Y,Q\n",
        encoding="utf-8",
    )
    data = base.model_dump(mode="json")
    data["identify_config"]["budget"]["max_candidates"] = 3
    data["executive_config"]["audit_options"]["max_tasks"] = 6
    data["executive_config"]["capabilities"]["available_metadata_fields"] = [
        "id", "label", "site", "device", "batch"
    ]
    request = RunRequest.model_validate(data)
    transport = FakeTransport()
    client = OpenAICompatibleClient(
        OpenAICompatibleModelConfig(
            model="offline", base_url="https://offline.invalid/v1",
            api_key_env="OFFLINE_KEY",
            model_call_budget={"max_total_calls": 40, "max_calls_per_agent": 20},
        ),
        api_key="offline", transport=transport, runs_root=runs_root,
    )
    registry = build_default_tool_registry(experiment_executor=RealExperimentExecutor())
    runtimes = {
        agent: ToolRuntime(
            registry, run_id=run_id, agent_name=agent,
            dataset_root=dataset_root, runs_root=runs_root,
            budget=ToolBudget(max_tool_calls=20, timeout_seconds=30),
        )
        for agent in ("quantitative", "qualitative")
    }

    def interrupt(_state, **_kwargs):
        raise RuntimeError("planned checkpoint interruption")

    original = getattr(graph_module, f"{interrupted_node}_node")
    monkeypatch.setattr(graph_module, f"{interrupted_node}_node", interrupt)
    with pytest.raises(RuntimeError, match="planned checkpoint interruption"):
        run(request, agent_client=client, tool_runtimes=runtimes)
    before = _agent_counts(transport)
    prior_tools = {
        path.relative_to(runs_root / run_id).as_posix()
        for path in (runs_root / run_id / "tools").glob("**/*.json")
    }
    monkeypatch.setattr(graph_module, f"{interrupted_node}_node", original)
    resumed = resume_run(run_id, runs_root=runs_root, agent_client=client,
                         tool_runtimes=runtimes)
    assert resumed.status == "COMPLETED"
    after = _agent_counts(transport)
    for agent, count in before.items():
        assert after[agent] == count
    later_tools = {
        path.relative_to(runs_root / run_id).as_posix()
        for path in (runs_root / run_id / "tools").glob("**/*.json")
    }
    assert prior_tools <= later_tools
