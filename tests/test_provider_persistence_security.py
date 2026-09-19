"""Offline regression for the real-provider persistence security boundary."""

import json
import subprocess
import sys
from pathlib import Path


def test_provider_response_sentinels_never_reach_persistence(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase16_provider_security_preflight.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "passed"
    assert report["raw_provider_body_persistence"] is False
    assert report["sentinel_hits"] == {
        "raw_provider_body": 0,
        "reasoning": 0,
        "authorization": 0,
        "base64": 0,
    }
    assert report["sanitized_telemetry_persisted"] is True
    assert report["image_input_verified"] is True
    assert report["image_payload_hits"] == 0
    assert report["base64_prefix_hits"] == 0
    assert report["offline_transport_calls"] == 2
    assert report["provider_calls"] == 0
