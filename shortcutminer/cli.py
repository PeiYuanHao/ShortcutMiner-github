"""Command-line boundary; execution remains owned by Runner/Orchestrator."""

import argparse
import json
import sys
from pathlib import Path

from shortcutminer.experiment.loader import load_model_runtime_config, load_run_request
from shortcutminer.experiment.run_manager import generate_run_id
from shortcutminer.evaluation.pilot.readiness import real_provider_configuration_blockers
from shortcutminer.evaluation.pilot.runner import PilotRunner
from shortcutminer.evaluation.pilot.suite import load_pilot_config
from shortcutminer.evaluation.pilot.credentials import credential_readiness
from shortcutminer.inspection.semantic_provider import (
    OpenAICompatibleSemanticVisualInspector,
)
from shortcutminer.protocol.model import ModelRuntimeConfigSet
from shortcutminer.runner import build_local_tool_runtimes, resume_run, run
from shortcutminer.runtime.demo import DemoAgentClient, build_demo_request
from shortcutminer.runtime.openai_compatible import (
    ModelBudgetState,
    OpenAICompatibleClient,
)
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
    live.add_argument(
        "--semantic-model-config",
        help="Optional image-capable OpenAI-compatible model JSON/YAML",
    )

    resume = commands.add_parser("resume", help="Resume an existing persisted run")
    resume.add_argument("run_id")
    resume.add_argument("--runs-root", default="runs")
    resume.add_argument("--runtime", choices=["mock", "openai-compatible"], required=True)
    resume.add_argument("--model-config", help="Required for model API resume")
    resume.add_argument(
        "--semantic-model-config",
        help="Optional image-capable OpenAI-compatible model JSON/YAML",
    )

    pilot = commands.add_parser("pilot", help="Phase 16 real-provider development pilot")
    pilot_commands = pilot.add_subparsers(dest="pilot_command", required=True)
    for name, help_text in (
        ("dry-run", "Generate and freeze the six-challenge plan without API calls"),
        ("credentials", "Report configured credential variables as SET or MISSING"),
        ("preflight", "Verify configured real text and VLM providers"),
        ("identify-diagnostic", "Run one gated real Identify compatibility diagnostic"),
        ("qualitative-diagnostic", "Run one gated real Qualitative compatibility diagnostic"),
        ("quantitative-diagnostic", "Run one gated real Quantitative compatibility diagnostic"),
        ("legacy-task-negative-replay", "Verify the persisted malformed task is rejected without providers"),
        ("regenerate-task-offline", "Regenerate and execute a task from a public CandidateReport without providers"),
        ("canary", "Run one real-provider end-to-end canary"),
        ("run", "Run the frozen scientific suite sequentially"),
        ("resume", "Continue only unattempted runs in the frozen suite"),
        ("report", "Regenerate report.md from the saved result"),
    ):
        command = pilot_commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="PilotConfig JSON/YAML")
        if name in {
            "qualitative-diagnostic", "quantitative-diagnostic",
            "legacy-task-negative-replay", "regenerate-task-offline",
        }:
            command.add_argument(
                "--source-pilot-id", required=True,
                help="Prior failed canary whose persisted public input is reused",
            )
        if name == "qualitative-diagnostic":
            command.add_argument(
                "--recover-from-pilot-id",
                help="Materialize the gate from one completed diagnostic run",
            )
    return parser


def _text_client_arguments(config, *, runs_root: str | Path) -> dict:
    if isinstance(config, ModelRuntimeConfigSet):
        budget_state = ModelBudgetState()
        return {
            f"{agent}_client": OpenAICompatibleClient(
                config.resolve(agent), runs_root=runs_root, budget_state=budget_state
            )
            for agent in (
                "identify",
                "executive",
                "quantitative",
                "qualitative",
                "summarize",
            )
        }
    return {"agent_client": OpenAICompatibleClient(config, runs_root=runs_root)}


def _semantic_inspector(config_path: str | None, *, runs_root: str | Path):
    if not config_path:
        return None
    config = load_model_runtime_config(config_path)
    if isinstance(config, ModelRuntimeConfigSet):
        raise ValueError("Semantic model configuration must contain one model runtime.")
    client = OpenAICompatibleClient(config, runs_root=runs_root)
    return OpenAICompatibleSemanticVisualInspector(client)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "pilot":
            config = load_pilot_config(args.config)
            repository_root = Path(config.repository_root).resolve(strict=True)
            artifact_root = Path(config.artifact_root)
            if not artifact_root.is_absolute():
                artifact_root = repository_root / artifact_root
            pilot_root = artifact_root.resolve(strict=False) / config.pilot_id
            needs_provider = args.pilot_command in {
                "preflight", "identify-diagnostic", "qualitative-diagnostic",
                "quantitative-diagnostic", "canary", "run", "resume"
            }
            if (
                args.pilot_command == "qualitative-diagnostic"
                and args.recover_from_pilot_id is not None
            ):
                needs_provider = False
            text_client = None
            semantic_inspector = None
            if needs_provider:
                blockers = real_provider_configuration_blockers(config)
                if blockers:
                    raise ValueError(
                        "real provider configuration is incomplete: "
                        + ", ".join(blockers)
                    )
                persistence_root = None if args.pilot_command == "preflight" else pilot_root / "runs"
                text_client = OpenAICompatibleClient(
                    config.text_runtime, runs_root=persistence_root,
                )
                semantic_client = OpenAICompatibleClient(
                    config.semantic_runtime, runs_root=persistence_root,
                )
                semantic_inspector = OpenAICompatibleSemanticVisualInspector(semantic_client)
            pilot_runner = PilotRunner(
                config, text_client=text_client, semantic_inspector=semantic_inspector,
            )
            if args.pilot_command == "dry-run":
                value = pilot_runner.dry_run()
            elif args.pilot_command == "credentials":
                value = {"credentials": credential_readiness(config)}
            elif args.pilot_command == "preflight":
                value = pilot_runner.preflight().model_dump(mode="json")
            elif args.pilot_command == "identify-diagnostic":
                value = pilot_runner.identify_diagnostic()
            elif args.pilot_command == "qualitative-diagnostic":
                value = pilot_runner.qualitative_diagnostic(
                    source_pilot_id=args.source_pilot_id,
                    recovery_pilot_id=args.recover_from_pilot_id,
                )
            elif args.pilot_command == "quantitative-diagnostic":
                value = pilot_runner.quantitative_diagnostic(
                    source_pilot_id=args.source_pilot_id,
                )
            elif args.pilot_command == "legacy-task-negative-replay":
                value = pilot_runner.legacy_task_negative_replay(
                    source_pilot_id=args.source_pilot_id,
                )
            elif args.pilot_command == "regenerate-task-offline":
                value = pilot_runner.regenerate_task_offline(
                    source_pilot_id=args.source_pilot_id,
                )
            elif args.pilot_command == "canary":
                value = pilot_runner.canary().model_dump(mode="json")
            elif args.pilot_command == "run":
                value = pilot_runner.run(resume=False).model_dump(mode="json")
            elif args.pilot_command == "resume":
                value = pilot_runner.run(resume=True).model_dump(mode="json")
            else:
                value = {"report": str(pilot_runner.report())}
            print(json.dumps({
                "event": "completed", "command": f"pilot {args.pilot_command}",
                "result": value,
            }), flush=True)
            return 0
        runtime_args = {}
        is_mock = False
        if args.command == "mock":
            client = DemoAgentClient(fail_at=args.fail_at)
            request = build_demo_request(args.runs_root, args.run_id or generate_run_id())
            runtime_args["agent_client"] = client
            is_mock = True
        elif args.command == "run":
            request = load_run_request(args.config)
            config = (
                load_model_runtime_config(args.model_config)
                if args.model_config else request.model_runtime
            )
            if config is None:
                raise ValueError("API run requires model_runtime or --model-config.")
            request = request.model_copy(update={"model_runtime": config})
            if request.run_id is None:
                request = request.model_copy(update={"run_id": generate_run_id()})
            runtime_args.update(
                _text_client_arguments(config, runs_root=request.runs_root)
            )
            semantic = _semantic_inspector(
                args.semantic_model_config, runs_root=request.runs_root
            )
            if semantic is not None:
                runtime_args["semantic_inspector"] = semantic
        elif args.runtime == "mock":
            if args.model_config:
                raise ValueError("Mock resume must not select a model configuration.")
            if args.semantic_model_config:
                raise ValueError("Mock resume must not select a semantic model configuration.")
            manifest = load_run_manifest(args.runs_root, args.run_id)
            if manifest.metadata.get("synthetic_demo") is not True:
                raise ValueError("Mock CLI resume is restricted to synthetic demo runs.")
            client = DemoAgentClient()
            runtime_args["agent_client"] = client
            is_mock = True
        else:
            if not args.model_config:
                raise ValueError("API resume requires --model-config.")
            config = load_model_runtime_config(args.model_config)
            runtime_args.update(
                _text_client_arguments(config, runs_root=args.runs_root)
            )
            semantic = _semantic_inspector(
                args.semantic_model_config, runs_root=args.runs_root
            )
            if semantic is not None:
                runtime_args["semantic_inspector"] = semantic

        print(json.dumps({"event": "started", "command": args.command}), flush=True)
        if args.command == "resume":
            if not is_mock:
                manifest = load_run_manifest(args.runs_root, args.run_id)
                runtime_args["tool_runtimes"] = build_local_tool_runtimes(
                    run_id=args.run_id,
                    dataset_root=manifest.dataset_info.root,
                    runs_root=args.runs_root,
                    approved_artifacts={
                        *manifest.artifacts.values(),
                        *manifest.config.get("executive_config", {}).get(
                            "artifact_references", []
                        ),
                    },
                )
            result = resume_run(
                args.run_id,
                runs_root=args.runs_root,
                **runtime_args,
            )
            runs_root = args.runs_root
        else:
            if not is_mock:
                runtime_args["tool_runtimes"] = build_local_tool_runtimes(
                    run_id=request.run_id,
                    dataset_root=request.dataset_info.root,
                    runs_root=request.runs_root,
                    approved_artifacts=set(
                        request.executive_config.get("artifact_references", [])
                    ),
                )
            result = run(request, **runtime_args)
            runs_root = request.runs_root
        summary = {
            "event": "completed",
            "run_id": result.run_id,
            "status": result.status,
            "final_result": str(Path(runs_root) / result.run_id / "final_result.json"),
        }
        if is_mock:
            summary["mock_agents_invoked"] = [call["agent_name"] for call in client.calls]
        print(json.dumps(summary), flush=True)
        return 0
    except Exception as exc:
        # Report failure and preserve a nonzero exit status; Runner owns durable recovery.
        print(json.dumps({"event": "failed", "error_type": type(exc).__name__,
                          "message": str(exc)}), file=sys.stderr, flush=True)
        return 1


__all__ = ["main"]
