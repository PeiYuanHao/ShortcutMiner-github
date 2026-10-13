import csv
from pathlib import Path

import numpy as np
import pytest

from shortcutminer.experiment.waterbirds_paper import (
    inspect_dataset, load_config, load_state, metrics, save_state, sha,
)


def test_frozen_matrix_is_two_models_three_seeds():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/experiments/waterbirds-paper-v1.json")
    assert config["models"] == ["resnet18", "resnet50"]
    assert config["seeds"] == [42, 43, 44]
    assert config["checkpoint_selection"] == "validation_overall_accuracy_then_earliest_epoch"


def test_group_metrics_use_training_weights_and_preserve_seed_uncertainty():
    result = metrics([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1],
                     train_group_counts={"0:0": 8, "0:1": 1, "1:0": 1, "1:1": 8}, seed=42)
    assert result["accuracy"] == result["balanced_accuracy"] == 0.5
    assert result["worst_group_accuracy"] == 0.0
    assert result["adjusted_average_accuracy"] == pytest.approx(16/18)
    assert result["groups"]["1:0"]["count"] == 1
    assert result["bootstrap"]["iterations"] == 1000
    assert "not_causal" in result["claim_scope"]


def test_missing_group_is_rejected_not_silently_dropped():
    with pytest.raises(ValueError, match="All four"):
        metrics([0, 1], [0, 1], [0, 1], train_group_counts={}, seed=42)


def test_path_traversal_fails_before_formal_training(tmp_path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not an image")
    root = tmp_path / "data"; root.mkdir()
    metadata = root / "metadata.csv"
    metadata.write_text("img_id,img_filename,y,place,split\n1,../outside.jpg,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside dataset root"):
        inspect_dataset({"dataset_root": str(root), "metadata_file": "metadata.csv", "expected_metadata_sha256": sha(metadata)})


def test_safe_checkpoint_roundtrip_and_tamper_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(2, 3)).sum().backward(); optimizer.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "weights.npz"
    digest = save_state(path, model, optimizer)
    restored = torch.nn.Linear(3, 2)
    other = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    load_state(path, digest, restored, other)
    assert all(torch.equal(expected[k], v) for k, v in restored.state_dict().items())
    assert all("momentum_buffer" in other.state[p] for p in restored.parameters())
    with np.load(path, allow_pickle=False) as data:
        assert all(data[k].dtype.kind in "biufc" for k in data.files)
    with path.open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_state(path, digest, restored)


def test_source_archive_cannot_be_relabelled_as_new_source(tmp_path, monkeypatch):
    import shortcutminer.experiment.waterbirds_paper as experiment
    import scripts.formal_environment as environment
    source = tmp_path / "model.py"
    source.write_text("original training source")
    state = {"source_tree_sha256": "first", "files": [{"path": "model.py", "sha256": sha(source)}]}
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(environment, "source_identity", lambda: state)
    archive = tmp_path / "source.zip"
    digest = experiment.archive_source(archive, "first")
    assert experiment.archive_source(archive, "first") == digest
    source.write_text("different source")
    state.update(source_tree_sha256="second", files=[{"path": "model.py", "sha256": sha(source)}])
    with pytest.raises(RuntimeError, match="Archived source identity changed"):
        experiment.archive_source(archive, "second")
    assert sha(archive) == digest
