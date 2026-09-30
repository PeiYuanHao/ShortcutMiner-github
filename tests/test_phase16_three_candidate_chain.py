"""Three-candidate offline run through the production graph and CPU backend."""

import json
from pathlib import Path

from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.run import RunRequest
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.runner import resume_run, run
from shortcutminer.runtime.demo import DemoAgentClient, build_demo_request
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime


class ThreeCandidateReasoning(DemoAgentClient):
    def _identify(self, payload):
        return {
            "candidates": [
                {
                    "cue": {"family": "metadata", "name": field},
                    "description": f"{field} may encode the label.",
                    "observation": f"{field} co-varies with label in public metadata.",
                    "hypothesis": f"The model may rely on {field}.",
                    "evidence_sources": ["metadata"],
                    "priority": "high",
                    "metadata_field_ref": field,
                    "additional_limitations": [],
                }
                for field in ("site", "device", "batch")
            ],
            "additional_limitations": [],
        }

    def _qualitative(self, payload):
        result = super()._qualitative(payload)
        field = payload["request"]["task"]["inputs"]["metadata_fields"][0]
        result["evidence_summary"] = f"Public metadata contains {field}."
        result["metadata_evidence"][0]["field"] = field
        return result

    def _summarize(self, payload):
        request = payload["request"]
        findings = []
        for entry in request["evidence_ledger"]["candidates"]:
            consistency = entry["profile"]["consistency"]
            assessment = {
                "mixed": "partially_supported",
                "inconclusive": "inconclusive",
                "no_usable_evidence": "insufficient_evidence",
                "consistent_contradiction": "contradicted",
            }.get(consistency, "supported")
            findings.append({
                "candidate_ref": entry["candidate_id"],
                "assessment": assessment,
                "summary": "Evidence was evaluated for this candidate alone.",
                "additional_limitations": [],
                "recommendations": ["Validate on independent public data."],
            })
        return {"overview": "Three candidate-specific evidence chains completed.",
                "findings": findings, "additional_limitations": []}


class FakeTransport:
    def __init__(self):
        self.reasoning = ThreeCandidateReasoning()
        self.calls = []

    def post_json(self, _url, *, headers, body, timeout):
        self.calls.append(body)
        payload = json.loads(body["messages"][-1]["content"])
        output = getattr(self.reasoning, f"_{payload['request']['target']}")(payload)
        return {
            "choices": [{"message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8,
                      "total_tokens": 18},
        }


def test_three_candidate_production_chain_and_resume(tmp_path: Path):
    runs_root = tmp_path / "runs"
    base = build_demo_request(runs_root, "phase16-three-candidate")
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
            registry, run_id=request.run_id, agent_name=agent,
            dataset_root=dataset_root, runs_root=runs_root,
            budget=ToolBudget(max_tool_calls=20, timeout_seconds=30),
        )
        for agent in ("quantitative", "qualitative")
    }
    execution = run(request, agent_client=client, tool_runtimes=runtimes)
    state = execution.workflow_state
    summary = SummaryReport.model_validate(state["final_result"])
    ledger = EvidenceLedger.model_validate(state["evidence_ledger"])
    assert len(summary.payload.findings) == len(ledger.candidates) == 3
    assert {entry.candidate_id for entry in ledger.candidates} == {
        "C001", "C002", "C003"
    }
    measurements = []
    for result in state["quantitative_result"]["results"]:
        payload = result["payload"]
        assert len(payload["measurements"]) == 1
        measurement = payload["measurements"][0]
        assert measurement["plan_step_id"]
        assert measurement["tool_call_id"]
        assert measurement["artifact_references"]
        assert measurement["task_id"] == payload["task_id"]
        assert measurement["candidate_id"] == payload["candidate_id"]
        measurements.append(measurement)
    ledger_measurements = [
        record for entry in ledger.candidates for record in entry.records
        if record.source_branch == "quantitative"
    ]
    assert len(ledger_measurements) == len(measurements)
    assert {
        (record.provenance.measurement_id, record.provenance.plan_step_id,
         record.provenance.tool_call_id)
        for record in ledger_measurements
    } == {
        (item["measurement_id"], item["plan_step_id"], item["tool_call_id"])
        for item in measurements
    }
    first_calls = len(transport.calls)
    resume_run(request.run_id, runs_root=runs_root, agent_client=client,
               tool_runtimes=runtimes)
    assert len(transport.calls) == first_calls
