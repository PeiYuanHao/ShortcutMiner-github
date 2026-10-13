"""Native Windows conda deployment; no scientific workflow modifications."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/environment/formal_experiment_environment.json"
RUNS = ROOT / "runs"
EXPERIMENTS = Path("E:/ShortcutMiner-Experiments")
CONFIGS = ROOT / "configs"
IDENTITY_FILES = ["requirements.txt", "requirements/core.in", "pyproject.toml",
    "requirements/formal-win64-py311.lock", "requirements/build-tools-win64.lock",
    "requirements/conda-win64.explicit.txt", "scripts/formal_environment.py",
    "scripts/formal_environment_lock.py", "scripts/phase17_progress_acceptance.py",
    "docs/formal-experiment-environment.md", "docs/deployment.md", "README.md"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(name):
    return name.lower().replace("_", "-").replace(".", "-")


def require_conda():
    if (sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}
            or platform.python_version() != "3.11.16"
            or Path(sys.prefix).name != "shortcutminer"
            or not (Path(sys.prefix) / "conda-meta").is_dir()):
        raise RuntimeError("Use native Windows conda shortcutminer, Python 3.11.16 AMD64")


def source_identity():
    raw = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=ROOT)
    names = sorted(set(raw.decode().split("\0")) - {""})
    records = []
    for name in names:
        if name.startswith("configs/environment/") or Path(name).name.startswith(".env"):
            continue
        path = ROOT / name
        if path.is_file():
            records.append({"path": name, "sha256": sha(path)})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "source_tree_sha256": digest, "files": records,
            "hash_exclusions": ["ignored files", ".env*", "configs/environment/*"]}


def refresh():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["source"] = source_identity()
    data["environment_hashes"] = {name: sha(ROOT / name) for name in IDENTITY_FILES}
    data["config_hashes"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(CONFIGS.rglob("*"))
                            if p.is_file() and "environment" not in p.relative_to(CONFIGS).parts
                            and not p.name.startswith(".env")}
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Windows candidate manifest refreshed; source policy recorded; workload capacity must be checked separately")


def check(data):
    require_conda()
    runtime = data["execution_environment"]
    if Path(sys.prefix).resolve() != Path(runtime["prefix"]).resolve():
        raise RuntimeError("Frozen conda prefix changed")
    if sha(Path(sys.executable)) != runtime["python_executable_sha256"]:
        raise RuntimeError("Frozen Python executable changed")
    for group in ("environment_hashes", "config_hashes"):
        for name, expected in data[group].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Frozen file changed: " + name)
    if source_identity()["source_tree_sha256"] != data["source"]["source_tree_sha256"]:
        raise RuntimeError("Source changed; review and refresh required")
    installed = {canonical(d.metadata["Name"]): d.version for d in metadata.distributions()}
    frozen_names = {package["name"] for group in ("dependencies", "build_tool_dependencies")
                    for package in data["resolution"][group]}
    unexpected = set(installed) - frozen_names - {"shortcutminer"}
    if unexpected:
        raise RuntimeError("Unfrozen packages installed: " + ", ".join(sorted(unexpected)))
    for group in ("dependencies", "build_tool_dependencies"):
        for package in data["resolution"][group]:
            if installed.get(package["name"]) != package["version"]:
                raise RuntimeError("Frozen package missing or changed: " + package["name"])
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, cwd=ROOT)


def offline_environment(data):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("SHORTCUTMINER_REAL_") or (name.startswith("SHORTCUTMINER_") and name.endswith("_API_KEY")) or name in {
            "SHORTCUTMINER_TEXT_API_KEY", "SHORTCUTMINER_VLM_API_KEY",
            "SHORTCUTMINER_MODEL_CONFIG", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"}:
            env.pop(name, None)
    env.update({name: str(value) for name, value in data["thread_limits"].items()})
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false",
               SHORTCUTMINER_DISABLE_FORMAL_ENV="1")
    return env


def run_bounded(command, env, seconds):
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        result = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        # Only this spawned process tree is terminated; never search by process name.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        raise RuntimeError("Acceptance timed out; no PASS recorded")
    if result:
        raise RuntimeError("Acceptance failed with exit " + str(result))


def finalize(data):
    identity = source_identity()
    snapshot_authorized = data.get("source_policy", {}).get("mode") == "user_authorized_source_snapshot"
    if identity["dirty"] and not snapshot_authorized:
        raise RuntimeError("Formal Windows runtime requires a user-created clean Git commit")
    receipt = {**data, "source": identity, "tracked_manifest_sha256": sha(MANIFEST),
               "status": "windows_runtime_source_snapshot_capacity_pending" if identity["dirty"] else "clean_windows_runtime_capacity_pending"}
    directory = RUNS / "environment-freeze-windows"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "formal_experiment_environment.final.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Windows runtime source identity recorded; workload capacity must be checked separately")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["refresh", "check", "pytest", "smoke", "finalize"])
    args = parser.parse_args()
    try:
        require_conda()
        if args.action == "refresh":
            refresh()
            return 0
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        check(data)
        if args.action == "check":
            print("Windows conda source/config/dependency identity: PASS")
        elif args.action == "finalize":
            finalize(data)
        else:
            env = offline_environment(data)
            if args.action == "pytest":
                command, seconds = [sys.executable, "-B", "-m", "pytest", "-q", "-ra"], 1800
            else:
                run_id = "win-offline-smoke-" + uuid.uuid4().hex[:12]
                command = [sys.executable, "-B", "scripts/phase17_progress_acceptance.py",
                           "--runs-root", str(RUNS), "--run-id", run_id]
                seconds = 300
            run_bounded(command, env, seconds)
        return 0
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("BLOCKED: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
