"""Command-line boundary; execution remains owned by Runner/Orchestrator."""

import argparse
import json
import sys
from pathlib import Path

from shortcutminer.experiment.loader import load_model_runtime_config, load_run_request
from shortcutminer.experiment.run_manager import generate_run_id
from shortcutminer.runner import resume_run, run
from shortcutminer.runtime.demo import DemoAgentClient, build_demo_request
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.storage.run_store import load_run_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shortcutminer")
    commands = parser.add_subparsers(dest="command", required=True)

    mock = commands.add_parser("mock", help="Run the offline synthetic five-agent demo")
    mock.add_argument("--run-id")
    mock.add_argument("--runs-root", default="runs")
    mock.add_argument("--fail-at", choices=["summarize"], help="Simulate a resumable failure")

    live = commands.add_parser("run", help="Explicitly run a configured model API")
    live.add_argument("--config", required=True, help="RunRequest JSON/YAML")
    live.add_argument("--model-config", help="Override model_runtime using JSON/YAML")

    resume = commands.add_parser("resume", help="Resume an existing persisted run")
    resume.add_argument("run_id")
    resume.add_argument("--runs-root", default="runs")
    resume.add_argument("--runtime", choices=["mock", "openai-compatible"], required=True)
    resume.add_argument("--model-config", help="Required for model API resume")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "mock":
            client = DemoAgentClient(fail_at=args.fail_at)
            request = build_demo_request(args.runs_root, args.run_id or generate_run_id())
        elif args.command == "run":
            request = load_run_request(args.config)
            config = (
                load_model_runtime_config(args.model_config)
                if args.model_config else request.model_runtime
            )
            if config is None:
                raise ValueError("API run requires model_runtime or --model-config.")
            request = request.model_copy(update={"model_runtime": config})
            client = OpenAICompatibleClient(config)
        elif args.runtime == "mock":
            if args.model_config:
                raise ValueError("Mock resume must not select a model configuration.")
            manifest = load_run_manifest(args.runs_root, args.run_id)
            if manifest.metadata.get("synthetic_demo") is not True:
                raise ValueError("Mock CLI resume is restricted to synthetic demo runs.")
            client = DemoAgentClient()
        else:
            if not args.model_config:
                raise ValueError("API resume requires --model-config.")
            client = OpenAICompatibleClient(load_model_runtime_config(args.model_config))

        print(json.dumps({"event": "started", "command": args.command}), flush=True)
        if args.command == "resume":
            result = resume_run(args.run_id, runs_root=args.runs_root, agent_client=client)
            runs_root = args.runs_root
        else:
            result = run(request, agent_client=client)
            runs_root = request.runs_root
        summary = {
            "event": "completed",
            "run_id": result.run_id,
            "status": result.status,
            "final_result": str(Path(runs_root) / result.run_id / "final_result.json"),
        }
        if isinstance(client, DemoAgentClient):
            summary["mock_agents_invoked"] = [call["agent_name"] for call in client.calls]
        print(json.dumps(summary), flush=True)
        return 0
    except Exception as exc:
        # Report failure and preserve a nonzero exit status; Runner owns durable recovery.
        print(json.dumps({"event": "failed", "error_type": type(exc).__name__,
                          "message": str(exc)}), file=sys.stderr, flush=True)
        return 1


__all__ = ["main"]
