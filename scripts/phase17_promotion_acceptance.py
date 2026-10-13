"""Local public fixture + fake provider + real CPU gate acceptance.

The explicit numerical policy is test_only, never a paper experiment policy.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/phase17-step6-acceptance/verified")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "tests"))
    from test_staged_validation import staged_request, TwoCandidates
    from test_audit_progress import Transport, client
    from shortcutminer.runner import run
    from shortcutminer.experiment.audit_report import export_audit_report
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    request, runtimes, executor = staged_request(directory)
    transport = Transport(TwoCandidates())
    result = run(request, agent_client=client(request.runs_root, transport), tool_runtimes=runtimes)
    root = Path(request.runs_root) / request.run_id
    counts = Counter(executor.calls)
    gate = result.workflow_state["promotion_result"]
    checks = {
        "pre_screen_both_promoted": result.workflow_state["screening_result"]["promoted"] == 2,
        "weak_terminated": gate["results"][0]["decision"] == "terminated_low_signal",
        "strong_promoted": gate["results"][1]["decision"] == "promoted",
        "stage1_once_per_candidate": counts[("C001", "statistical_check")] == counts[("C002", "statistical_check")] == 1,
        "weak_expensive_calls_zero": all(candidate != "C001" or kind == "statistical_check" for candidate, kind in executor.calls),
        "strong_stage2_executed": counts[("C002", "model_evaluation")] == counts[("C002", "intervention")] == 1,
        "five_agents": set(transport.calls) == {"identify", "executive", "quantitative", "qualitative", "summarize"},
    }
    before = (root / "audit_report.json").read_bytes(), (root / "audit_report.md").read_bytes()
    report = export_audit_report(request.runs_root, request.run_id)
    checks["idempotent_report"] = before == ((root / "audit_report.json").read_bytes(), (root / "audit_report.md").read_bytes())
    checks["findings_preserved"] = report.findings == result.final_result["payload"]
    checks["report_safety"] = all(token not in report.model_dump_json() for token in ("api_key", "Authorization", "hidden_truth", "base64", "raw_provider"))
    evidence = {"acceptance": "PASS" if all(checks.values()) else "BLOCKED", "checks": checks,
        "promotion_policy_status": "test_only", "promotion_result": gate,
        "backend_calls": [{"candidate_id": candidate, "experiment_type": kind, "count": count} for (candidate, kind), count in sorted(counts.items())],
        "fake_provider_calls": len(transport.calls), "real_provider_calls": 0, "report_directory": str(root)}
    (directory / "acceptance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
