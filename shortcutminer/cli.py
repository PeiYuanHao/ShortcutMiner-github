"""Command-line boundary; execution remains owned by Runner/Orchestrator."""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from shortcutminer.audit import build_audit_run_request, persist_audit_request, reserve_audit_intake
from shortcutminer.protocol.audit import AuditLabelInformation, AuditRequest

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
from shortcutminer.runtime.formal_environment import command_uses_formal_credentials, formal_credentials
from shortcutminer.runtime.openai_compatible import (
    ModelBudgetState,
    OpenAICompatibleClient,
)
from shortcutminer.storage.run_store import load_run_manifest
from shortcutminer.workflow.observability import lifecycle, emit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shortcutminer")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Audit a dataset through the existing five-agent workflow")
    audit.add_argument("dataset_path", help="Dataset directory or tabular file")
    audit.add_argument("--description", help="Optional dataset description")
    audit.add_argument("--metadata-file", help="Optional exact metadata file relative to dataset root")
    audit.add_argument("--label-column", help="Optional exact metadata label column")
    audit.add_argument("--label-source", choices=["class_directory", "split_class_directory"])
    audit.add_argument("--run-id")
    audit.add_argument("--runs-root", default="runs")
    audit.add_argument("--model-config", help="Model settings (or SHORTCUTMINER_MODEL_CONFIG)")
    audit.add_argument("--semantic-model-config", help="Optional visual model settings")
    audit_progress = audit.add_mutually_exclusive_group()
    audit_progress.add_argument("--progress", dest="progress", action="store_true", help="Display live progress (default)")
    audit_progress.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress output")
    audit.set_defaults(progress=True)

    mock = commands.add_parser("mock", help="Run the offline synthetic five-agent demo")
    mock.add_argument("--run-id")
    mock.add_argument("--runs-root", default="runs")
    mock.add_argument("--fail-at", choices=["summarize"], help="Simulate a resumable failure")
    mock.add_argument("--progress", action="store_true", help="Display live event progress on stderr")

    live = commands.add_parser("run", help="Explicitly run a configured model API")
    live.add_argument("--config", required=True, help="RunRequest JSON/YAML")
    live.add_argument("--model-config", help="Override model_runtime using JSON/YAML")
    live.add_argument("--progress", action="store_true", help="Display live event progress on stderr")
    live.add_argument(
        "--semantic-model-config",
        help="Optional image-capable OpenAI-compatible model JSON/YAML",
    )

    resume = commands.add_parser("resume", help="Resume an existing persisted run")
    resume.add_argument("run_id")
    resume.add_argument("--runs-root", default="runs")
    resume.add_argument("--runtime", choices=["mock", "openai-compatible"], required=True)
    resume.add_argument("--progress", action="store_true", default=None, help="Replay saved progress and display live events on stderr")
    resume.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress output")
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
        ("historical-candidate-binding-replay", "Verify ambiguous historical public candidates remain unresolved"),
        ("current-contract-task-offline", "Execute the current exact-binding contract through the real local backend"),
        ("canary", "Run one real-provider end-to-end canary"),
        ("run", "Run the frozen scientific suite sequentially"),
        ("resume", "Continue only unattempted runs in the frozen suite"),
        ("report", "Regenerate report.md from the saved result"),
    ):
        command = pilot_commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="PilotConfig JSON/YAML")
        if name in {
            "qualitative-diagnostic", "quantitative-diagnostic",
            "legacy-task-negative-replay", "historical-candidate-binding-replay",
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
        with formal_credentials(enabled=command_uses_formal_credentials(args)):
            return _execute_command(args)
    except (OSError, ValueError):
        print(json.dumps({"event": "failed", "message": "Unable to load .env.formal; check local file syntax and access."}),
              file=sys.stderr, flush=True)
        return 1


def _execute_command(args) -> int:
    try:
        if args.command == "audit":
            return _audit_command(args)
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
            elif args.pilot_command == "historical-candidate-binding-replay":
                value = pilot_runner.historical_candidate_binding_replay(
                    source_pilot_id=args.source_pilot_id,
                )
            elif args.pilot_command == "current-contract-task-offline":
                value = pilot_runner.current_contract_task_offline()
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

        if args.command == "resume" and args.progress is None:
            args.progress = load_run_manifest(args.runs_root, args.run_id).metadata.get("entrypoint") == "audit"
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
                    restore_persisted_task_results=True,
                )
            with _progress_context(args.progress, args.runs_root, args.run_id):
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
            with _progress_context(args.progress, request.runs_root, request.run_id):
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
        if getattr(getattr(result, "manifest", None), "metadata", {}).get("entrypoint") == "audit":
            summary["audit_report"] = str(Path(runs_root) / result.run_id / "audit_report.md")
            summary["audit_report_json"] = str(Path(runs_root) / result.run_id / "audit_report.json")
            print(f"Audit completed.\nReport: {summary['audit_report']}\nMachine-readable: {summary['audit_report_json']}", file=sys.stderr)
        print(json.dumps(summary), flush=True)
        return 0
    except Exception as exc:
        # Report failure and preserve a nonzero exit status; Runner owns durable recovery.
        from shortcutminer.monitoring.safety import safe_text
        message = "execution_failed; artifacts preserved if intake started"
        if args.command == "mock" and args.fail_at == "summarize":
            message = "Planned mock failure at summarize."
        if args.command == "audit" and isinstance(exc, ValueError) and str(exc).startswith("Audit requires preconfigured"):
            message = "Audit requires preconfigured model settings: SHORTCUTMINER_MODEL_CONFIG or --model-config."
        print(json.dumps({"event": "failed", "error_type": safe_text(type(exc).__name__),
                          "message": message}), file=sys.stderr, flush=True)
        return 1


def _progress_context(enabled: bool, runs_root, run_id):
    if not enabled:
        return nullcontext()
    from shortcutminer.monitoring.renderer import ProgressMonitor
    return ProgressMonitor(runs_root, run_id)


def _audit_command(args):
    """CLI observation surrounds intake and Runner; no task or plan is generated here."""
    if args.label_column and args.label_source:
        raise ValueError("Select either --label-column or --label-source.")
    Path(args.dataset_path).expanduser().resolve(strict=True)
    config_path = args.model_config or os.environ.get("SHORTCUTMINER_MODEL_CONFIG")
    if not config_path:
        raise ValueError("Audit requires preconfigured model settings: SHORTCUTMINER_MODEL_CONFIG or --model-config.")
    config = load_model_runtime_config(config_path)
    runtime_args = _text_client_arguments(config, runs_root=args.runs_root)
    semantic = _semantic_inspector(args.semantic_model_config, runs_root=args.runs_root)
    if semantic is not None:
        runtime_args["semantic_inspector"] = semantic
    label_information = (AuditLabelInformation(source="metadata_column", column=args.label_column) if args.label_column
                         else AuditLabelInformation(source=args.label_source) if args.label_source else None)
    intake = AuditRequest(dataset_path=args.dataset_path, dataset_description=args.description,
                         metadata_file=args.metadata_file, label_information=label_information)
    run_id = args.run_id or generate_run_id()
    reserve_audit_intake(intake, runs_root=args.runs_root, run_id=run_id)
    print(json.dumps({"event": "started", "command": "audit"}), flush=True)
    with _progress_context(args.progress, args.runs_root, run_id):
        try:
            with lifecycle(args.runs_root, run_id, "audit_intake"):
                request = build_audit_run_request(intake, runs_root=args.runs_root, run_id=run_id, model_runtime=config)
                persist_audit_request(request, reserved_intake=intake)
            runtime_args["tool_runtimes"] = build_local_tool_runtimes(
                run_id=run_id, dataset_root=request.dataset_info.root, runs_root=request.runs_root,
                approved_artifacts=set(request.executive_config.get("artifact_references", [])))
            result = run(request, **runtime_args)
        except BaseException:
            emit(args.runs_root, run_id, "RUN_FAILED", "run", reason="stage_execution_failed")
            raise
    print(json.dumps({"event": "completed", "run_id": result.run_id, "status": result.status,
                      "final_result": str(Path(args.runs_root) / run_id / "final_result.json"),
                      "audit_request": str(Path(args.runs_root) / run_id / "audit_request.json"),
                      "audit_report": str(Path(args.runs_root) / run_id / "audit_report.md"),
                      "audit_report_json": str(Path(args.runs_root) / run_id / "audit_report.json")}), flush=True)
    print(f"Audit completed.\nReport: {Path(args.runs_root) / run_id / 'audit_report.md'}\n"
          f"Machine-readable: {Path(args.runs_root) / run_id / 'audit_report.json'}", file=sys.stderr)
    return 0


__all__ = ["main"]
