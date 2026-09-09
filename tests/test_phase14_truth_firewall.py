import hashlib
import json
from pathlib import Path

from PIL import Image

from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.experiment.visual_discovery import VisualDiscoveryProfiler
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.protocol.identify import (
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.visual_discovery import VisualDiscoveryOptions
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient


class PublicObservationTransport:
    def post_json(self, url, *, headers, body, timeout):
        public = json.loads(body["messages"][1]["content"][0]["text"])
        sample = public["samples_in_image_order"][0]
        output = {
            "schema_version": "1.0",
            "observations": [{
                "sample_id": sample["sample_id"],
                "sample_reference": sample["sample_reference"],
                "observation_type": "background_context",
                "region": "full image",
                "description": "uniform visible color field",
                "confidence": "high",
                "limitations": [],
            }],
        }
        return {"choices": [{"message": {"content": json.dumps(output)}}]}


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_truth_firewall_hashes(tmp_path: Path):
    challenge = tmp_path / "challenge"
    public = challenge / "public" / "dataset"
    private = challenge / "private"
    public.mkdir(parents=True)
    private.mkdir()
    Image.new("RGB", (10, 10), (40, 150, 60)).save(public / "sample.png")
    (public / "metadata.csv").write_text(
        "sample_id,sample_path,label\na,sample.png,0\n", encoding="utf-8"
    )
    truth_path = private / "hidden_truth.json"
    truth_path.write_text('{"answer":"present"}\n', encoding="utf-8")
    request = IdentifyRequest(
        protocol_version="1.0",
        run_id="phase14-firewall",
        message_id="phase14-firewall-request",
        source="orchestrator",
        target="identify",
        message_type="identify_request",
        dataset=DatasetSpec(
            name="neutral-public-dataset", root=str(public), metadata_file="metadata.csv"
        ),
        task=TaskSpec(type="classification", label_column="label"),
        budget=IdentifyBudget(
            max_samples=1, max_candidates=1, max_dataset_entries=20,
            max_metadata_rows=20,
        ),
    )
    dataset_profile = build_dataset_profile(request, seed=71)
    outputs = []
    original = truth_path.read_bytes()
    for state in ("present", "removed", "replaced"):
        if state == "removed":
            truth_path.unlink()
        elif state == "replaced":
            truth_path.write_text('{"answer":"different"}\n', encoding="utf-8")
        client = OpenAICompatibleClient(
            OpenAICompatibleModelConfig(
                model="fake-vision",
                base_url="https://vision.example/v1",
                api_key_env="SHORTCUTMINER_TEST_KEY",
                capabilities={"supports_image_input": True},
            ),
            api_key="fake-secret",
            transport=PublicObservationTransport(),
        )
        visual = VisualDiscoveryProfiler().profile(
            request,
            dataset_profile,
            seed=71,
            options=VisualDiscoveryOptions(
                max_samples=1,
                sample_id_column="sample_id",
                sample_path_column="sample_path",
                semantic_enabled=True,
            ),
            semantic_inspector=OpenAICompatibleSemanticVisualInspector(client),
        )
        outputs.append((
            client.invocations[0].payload_hash,
            client.invocations[0].request_hash,
            _hash(visual.model_dump(mode="json")),
        ))
    truth_path.write_bytes(original)
    return outputs


def test_truth_present_removed_replaced_keep_model_semantic_and_visual_hashes(tmp_path: Path):
    outputs = compute_truth_firewall_hashes(tmp_path)
    assert outputs[0] == outputs[1] == outputs[2]
