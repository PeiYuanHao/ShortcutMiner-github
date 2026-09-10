import hashlib
import json
from pathlib import Path

from test_model_reliance import (
    execute,
    evaluation_parameters,
    make_shape_dataset,
    runtime_for,
    train,
    training_parameters,
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def compute_phase15_truth_firewall_hashes(tmp_path: Path):
    outputs = []
    for state in ("present", "removed", "replaced"):
        root = tmp_path / state
        dataset = make_shape_dataset(root)
        private = root / "private"
        private.mkdir()
        truth = private / "hidden_truth.json"
        if state == "present":
            truth.write_text('{"target":"background color"}\n', encoding="utf-8")
        elif state == "replaced":
            truth.write_text('{"target":"unrelated corner"}\n', encoding="utf-8")
        runtime, runs = runtime_for(root, dataset)
        training_request = training_parameters()
        training, model_reference = train(runtime)
        intervention_request = {
            **evaluation_parameters(model_reference),
            "transformation": {"type": "global_color", "mode": "swap_red_blue"},
            "target_cue": "background color", "comparison_mode": "paired",
            "preservation_status": "guaranteed_by_construction",
            "max_intervention_samples": 20,
        }
        ood = execute(runtime, "ood_evaluation", intervention_request, f"ood-{state}")
        assert ood.status == "success"
        manifest_reference = next(
            ref for ref in ood.output["artifact_references"] if ref.endswith("intervention_manifest.json")
        )
        manifest = json.loads((runs / "phase15-run" / manifest_reference).read_text(encoding="utf-8"))
        routing_plan = {
            "candidate": {
                "candidate_id": "C001", "cue": {"family": "appearance", "name": "background color"},
                "observation": "Public visual evidence shows label-correlated backgrounds.",
            },
            "requested_checks": ["task_model_training", "ood_evaluation", "counterfactual_intervention"],
            "required_model_capabilities": ["retraining", "checkpoint", "intervention_inference"],
        }
        outputs.append({
            "routing_plan_hash": _hash(routing_plan),
            "model_training_request_hash": _hash(training_request),
            "intervention_request_hash": _hash(intervention_request),
            "ood_public_artifact_hash": _hash(manifest),
            "model_hash": training.output["metadata"]["details"]["model_hash"],
        })
    return outputs


def test_truth_present_removed_replaced_do_not_change_phase15_runtime_hashes(tmp_path: Path):
    outputs = compute_phase15_truth_firewall_hashes(tmp_path)
    assert outputs[0] == outputs[1] == outputs[2]
