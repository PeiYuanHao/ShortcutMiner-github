"""Read-only acceptance checks, run INSIDE a newly created container.

No Docker daemon or host Python is needed by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

from shortcutminer.storage.run_store import load_run_manifest, load_run_record
from shortcutminer.workflow.checkpoint import checkpoint_config, open_sqlite_checkpointer


def verify(runs_root: Path, run_id: str, *, failed: bool, resumed: bool) -> dict:
    record = load_run_record(runs_root, run_id)
    expected = "FAILED" if failed else "COMPLETED"
    assert record.status == expected, record.status
    assert load_run_manifest(runs_root, run_id).status == expected
    run_dir = runs_root / run_id
    paths = ["run.json", "manifest.json", "input.json", "events.jsonl",
             "workflow_state.json", "checkpoints.sqlite", "identify/output.json",
             "executive/routing.json", "quantitative/TASK-C001-QT/result.json",
             "qualitative/TASK-C001-QL/result.json"]
    if not failed:
        paths += ["summarize/result.json", "final_result.json"]
    for name in paths:
        assert (run_dir / name).is_file(), name
    with open_sqlite_checkpointer(run_dir / "checkpoints.sqlite") as saver:
        snapshot = saver.get_tuple(checkpoint_config(run_id))
        assert snapshot is not None
        assert snapshot.checkpoint["channel_values"]["run_id"] == run_id
        assert saver.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    events = [json.loads(line)["event"] for line in
              (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    for agent in ("IDENTIFY", "EXECUTIVE", "QUANTITATIVE", "QUALITATIVE"):
        assert events.count(f"{agent}_COMPLETED") == 1, (agent, events)
    if resumed:
        assert record.attempt_count == 2
        assert "RUN_FAILED" in events and "RUN_RESUMED" in events
        assert events.count("SUMMARIZE_COMPLETED") == 1
    if not failed:
        final = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))
        assert final["run_id"] == run_id
        assert final["message_type"] == "summary_report"
        assert len(final["payload"]["findings"]) == 1
    return {"verification": "PASS", "run_id": run_id, "status": expected,
            "attempt_count": record.attempt_count,
            "sha256": {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                       for name in paths if name != "checkpoints.sqlite"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--runs-root", type=Path, default=Path("/runs"))
    parser.add_argument("--failed", action="store_true")
    parser.add_argument("--resumed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.runs_root, args.run_id, failed=args.failed,
                            resumed=args.resumed), sort_keys=True))
