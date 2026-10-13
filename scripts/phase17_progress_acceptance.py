"""Manual audit CLI acceptance against a loopback-only fake provider.

Uses the installed shortcutminer command, real CPU backends, and fake model usage.
No external provider is contacted. Captured logs remain under the run directory.
"""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/phase17-step4-acceptance")
    parser.add_argument("--run-id", default="audit-progress-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "tests"))
    from test_audit_progress import AuditReasoning
    reasoning = AuditReasoning()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = next(json.loads(message["content"]) for message in body["messages"]
                           if message["role"] == "user" and message["content"].lstrip().startswith("{"))
            agent = payload["request"]["target"]
            calls.append(agent)
            output = getattr(reasoning, f"_{agent}")(payload)
            response = json.dumps({"choices": [{"message": {"content": json.dumps(output)}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with TemporaryDirectory(prefix="shortcutminer-fake-progress-") as directory:
            config = Path(directory) / "model.json"
            config.write_text(json.dumps({"model": "phase17-local-fake", "api_key_env": "PHASE17_FAKE_KEY",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1"}), encoding="utf-8")
            environment = {**os.environ, "SHORTCUTMINER_MODEL_CONFIG": str(config), "PHASE17_FAKE_KEY": "offline",
                           "SHORTCUTMINER_DISABLE_FORMAL_ENV": "1",
                           "PYTHONIOENCODING": "utf-8"}
            command = [sys.executable, "-m", "shortcutminer", "audit", "tests/fixtures/generic_dataset_a", "--runs-root", args.runs_root, "--run-id", args.run_id]
            result = subprocess.run(command, cwd=repository, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
            run_dir = repository / args.runs_root / args.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "progress.stdout.log").write_text(result.stdout, encoding="utf-8")
            (run_dir / "progress.stderr.log").write_text(result.stderr, encoding="utf-8")
            print(result.stderr, end="")
            print(result.stdout, end="")
            counts = [int(value) for value in re.findall(r"Text API calls=(\d+)", result.stderr)]
            passed = (result.returncode == 0 and len(calls) == 5 and "\x1b" not in result.stderr
                      and counts and counts[-1] == len(calls) and max(counts) <= len(calls))
            report_checks = {}
            if result.returncode == 0:
                from shortcutminer.experiment.audit_report import export_audit_report, render_audit_markdown
                from shortcutminer.protocol.audit_report import AuditReport
                report = AuditReport.model_validate_json((run_dir / "audit_report.json").read_text(encoding="utf-8"))
                state = json.loads((run_dir / "workflow_state.json").read_text(encoding="utf-8"))
                final = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))
                markdown = (run_dir / "audit_report.md").read_text(encoding="utf-8")
                before = (run_dir / "audit_report.json").read_bytes(), (run_dir / "audit_report.md").read_bytes()
                export_audit_report(repository / args.runs_root, args.run_id)
                content = report.model_dump_json() + markdown
                report_checks = {
                    "same_canonical_markdown": markdown == render_audit_markdown(report),
                    "dataset": report.dataset["name"] == "generic_dataset_a" and report.dataset["sample_count"] == 4,
                    "all_candidates": len(report.discovery["candidates"]) == len(state["identify_result"]["candidates"]) == 2,
                    "screening": report.screening["promoted"] == 2 and report.screening["screened_out"] == 0,
                    "tasks": len(report.validation["routing"]["tasks"]) == 2,
                    "evidence": report.evidence == state["evidence_ledger"],
                    "findings": report.findings == final["payload"],
                    "cost": report.cost["total_tokens"] == 90 and report.cost["text_provider_calls"] == 5,
                    "reproducibility": report.reproducibility["seed"] == 42 and bool(report.runtime["snapshots"]),
                    "security": all(word not in content for word in ("Authorization", "api_key", "PHASE17_FAKE_KEY", "hidden_truth", "base64", "raw_provider_body")),
                    "idempotence": before == ((run_dir / "audit_report.json").read_bytes(), (run_dir / "audit_report.md").read_bytes()),
                }
                passed = passed and all(report_checks.values())
            print(json.dumps({"acceptance": "PASS" if passed else "BLOCKED", "fake_provider_calls": len(calls),
                              "real_provider_calls": 0, "report_checks": report_checks, "artifacts": str(run_dir)}))
            return 0 if passed else 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
