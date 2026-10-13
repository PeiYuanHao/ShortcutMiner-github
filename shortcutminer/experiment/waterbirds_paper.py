"""Explicit image-model experiment matrix, separate from scientific Agent decisions.

No provider calls, label-driven intervention, or fabricated EvidenceLedger entries.
Pretrained weights are allowlisted Torchvision tensor dictionaries. Run checkpoints
are non-executable numeric NPZ files with hash-checked JSON provenance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import zipfile
import importlib.metadata

import numpy as np
from PIL import Image

from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.storage.json_artifacts import atomic_write_json


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("resnet18", "resnet50")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    atomic_write_json(Path(path), value, error_type=RuntimeError)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["models"] != list(MODELS) or config["seeds"] != [42, 43, 44]:
        raise ValueError("Frozen matrix must be resnet18/resnet50 x seeds 42/43/44")
    for field in ("epochs", "batch_size", "checkpoint_interval", "torch_threads", "bootstrap_iterations"):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] < 1:
            raise ValueError("Invalid positive integer: " + field)
    if config["workers"] < 0 or not 0 < config["confidence_level"] < 1:
        raise ValueError("Invalid workers or confidence level")
    if config["checkpoint_selection"] != "validation_overall_accuracy_then_earliest_epoch":
        raise ValueError("Test-based model selection is forbidden")
    return config


def source_identity(*, require_clean):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if require_clean and dirty:
        raise RuntimeError("Formal training requires your manual clean Git commit; no training started")
    from scripts.formal_environment import source_identity as snapshot
    return {"git_commit": commit, "dirty": dirty, "source_tree_sha256": snapshot()["source_tree_sha256"]}


def archive_source(path, expected_source_hash):
    from scripts.formal_environment import source_identity as snapshot
    source = snapshot()
    if source["source_tree_sha256"] != expected_source_hash:
        raise RuntimeError("Source changed while freezing matrix")
    path = Path(path)
    if path.exists():
        # A reused directory must never describe new source with an old archive.
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != [record["path"] for record in source["files"]]:
                raise RuntimeError("Archived source identity changed; use a new output directory")
            for record in source["files"]:
                if hashlib.sha256(archive.read(record["path"])).hexdigest() != record["sha256"]:
                    raise RuntimeError("Archived source identity changed; use a new output directory")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for record in source["files"]:
                payload = (ROOT / record["path"]).read_bytes()
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise RuntimeError("Source changed while archiving")
                info = zipfile.ZipInfo(record["path"], date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)
        temporary.replace(path)
    return sha(path)


def inspect_dataset(config, *, decode=False):
    root = Path(config["dataset_root"]).resolve(strict=True)
    metadata = (root / config["metadata_file"]).resolve(strict=True)
    if not metadata.is_relative_to(root) or sha(metadata) != config["expected_metadata_sha256"]:
        raise ValueError("Metadata escaped root or changed")
    rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig", newline="")))
    ids, paths, records = set(), set(), []
    for row in rows:
        if row["img_id"] in ids or row["img_filename"] in paths:
            raise ValueError("Duplicate sample identity or filename")
        ids.add(row["img_id"])
        paths.add(row["img_filename"])
        path = (root / row["img_filename"]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Image must be a real file inside dataset root")
        if row["y"] not in {"0", "1"} or row["place"] not in {"0", "1"} or row["split"] not in {"0", "1", "2"}:
            raise ValueError("Unexpected label, evaluation group, or official split")
        if decode:
            load_rgb_image_file(path)
        records.append({"img_id": row["img_id"], "img_filename": row["img_filename"],
                        "y": row["y"], "place": row["place"], "split": row["split"], "sha256": sha(path)})
    counts = {split: sum(r["split"] == split for r in rows) for split in ("0", "1", "2")}
    if counts != {"0": 4795, "1": 1199, "2": 5794}:
        raise ValueError("Official Waterbirds split counts changed")
    return rows, {"metadata_sha256": sha(metadata), "split_counts": counts,
                  "image_count": len(records), "image_manifest": records,
                  "data_sha256": identity(records), "all_images_decoded": decode}


def metrics(labels, groups, predictions, *, train_group_counts, seed, iterations=1000, confidence=0.95):
    y, g, pred = [np.asarray(v, dtype=np.int64) for v in (labels, groups, predictions)]
    if not len(y) or len(y) != len(g) or len(y) != len(pred):
        raise ValueError("Prediction/label/group rows must be aligned and nonempty")
    if any(np.any(~np.isin(v, [0, 1])) for v in (y, g, pred)):
        raise ValueError("Binary labels, groups, and predictions required")
    correct = (y == pred).astype(float)
    keys = [f"{label}:{group}" for label in (0, 1) for group in (0, 1)]
    strata = [correct[(y == label) & (g == group)] for label in (0, 1) for group in (0, 1)]
    if any(not len(values) for values in strata):
        raise ValueError("All four evaluation groups must be present")
    values = np.array([s.mean() for s in strata])
    counts = np.array([len(s) for s in strata])
    weights = np.array([train_group_counts[k] for k in keys], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid training group weights")
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    boot = np.column_stack([rng.binomial(len(s), s.mean(), size=iterations) / len(s) for s in strata])
    alpha = (1 - confidence) / 2
    interval = lambda v: [float(x) for x in np.quantile(v, [alpha, 1 - alpha])]
    balanced = float(np.mean([correct[y == label].mean() for label in (0, 1)]))
    return {"sample_count": len(y), "accuracy": float(correct.mean()), "balanced_accuracy": balanced,
            "adjusted_average_accuracy": float(values @ weights), "worst_group_accuracy": float(values.min()),
            "groups": {k: {"count": int(n), "accuracy": float(v), "ci": interval(boot[:, i])}
                       for i, (k, n, v) in enumerate(zip(keys, counts, values))},
            "accuracy_ci": interval(boot @ (counts / counts.sum())),
            "adjusted_average_accuracy_ci": interval(boot @ weights),
            "worst_group_accuracy_ci": interval(boot.min(axis=1)),
            "bootstrap": {"method": "within_group_binary_prediction_bootstrap", "iterations": iterations,
                          "confidence_level": confidence, "seed": seed},
            "claim_scope": "model_group_performance_not_causal_confirmation"}


def torch_runtime(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA matrix requires an NVIDIA GPU; no CPU fallback")
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def seed_worker(_):
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class ImageRows:
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["img_filename"]).resolve(strict=True)
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Image escaped dataset root")
        decoded = load_rgb_image_file(path)
        return self.transform(Image.fromarray(decoded.rgb)), int(row["y"]), int(row["place"]), index


def transforms(training):
    from torchvision import transforms as T
    spatial = [T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip()] if training else [T.Resize(256), T.CenterCrop(224)]
    return T.Compose([*spatial, T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def loader(config, rows, *, training, seed):
    import torch
    return torch.utils.data.DataLoader(ImageRows(config["dataset_root"], rows, transforms(training)),
        batch_size=config["batch_size"], shuffle=training, num_workers=config["workers"],
        pin_memory=True, generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
        persistent_workers=False)


def initialization(architecture):
    import torch
    from torchvision.models import ResNet18_Weights, ResNet50_Weights
    enum = ResNet18_Weights.IMAGENET1K_V1 if architecture == "resnet18" else ResNet50_Weights.IMAGENET1K_V1
    path = Path(torch.hub.get_dir()) / "checkpoints" / enum.url.rsplit("/", 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = path.stem.rsplit("-", 1)[1]
    if not path.exists():
        torch.hub.download_url_to_file(enum.url, str(path), hash_prefix=prefix, progress=False)
    if not sha(path).startswith(prefix):
        raise RuntimeError("Official initialization checksum mismatch")
    return path, {"architecture": architecture, "url": enum.url, "sha256": sha(path), "weights": enum.name}


def build_model(architecture, weights_path):
    import torch
    from torchvision.models import resnet18, resnet50
    constructor = {"resnet18": resnet18, "resnet50": resnet50}[architecture]
    model = constructor(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model.cuda()


def save_state(path, model, optimizer=None):
    """Only numeric tensors; never pickle/module serialization."""
    arrays = {"model/" + name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    if optimizer is not None:
        for name, parameter in model.named_parameters():
            momentum = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if momentum is not None:
                arrays["momentum/" + name] = momentum.detach().cpu().numpy()
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
    temp.replace(path)
    return sha(path)


def load_state(path, expected_hash, model, optimizer=None):
    import torch
    if sha(path) != expected_hash:
        raise RuntimeError("Checkpoint hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        expected = model.state_dict()
        if {k[6:] for k in arrays.files if k.startswith("model/")} != set(expected):
            raise RuntimeError("Checkpoint tensor names changed")
        state = {}
        for name, value in expected.items():
            array = arrays["model/" + name]
            if array.shape != tuple(value.shape) or not np.isfinite(array).all():
                raise RuntimeError("Checkpoint shape or finite-value mismatch")
            state[name] = torch.from_numpy(array.copy()).to(value.device, dtype=value.dtype)
        model.load_state_dict(state, strict=True)
        if optimizer is not None:
            for name, parameter in model.named_parameters():
                key = "momentum/" + name
                if key in arrays:
                    array = arrays[key]
                    if array.shape != tuple(parameter.shape) or not np.isfinite(array).all():
                        raise RuntimeError("Invalid optimizer momentum")
                    optimizer.state[parameter]["momentum_buffer"] = torch.from_numpy(array.copy()).to(parameter.device)


def predict(model, data_loader):
    import torch
    model.eval()
    labels, groups, predicted, probabilities, indices = [], [], [], [], []
    with torch.inference_mode():
        for images, y, g, index in data_loader:
            probs = model(images.cuda(non_blocking=True)).float().softmax(dim=1).cpu()
            labels.extend(y.tolist()); groups.extend(g.tolist())
            predicted.extend(probs.argmax(1).tolist()); probabilities.extend(probs.tolist()); indices.extend(index.tolist())
    return labels, groups, predicted, probabilities, indices


def capacity(config, output):
    torch = torch_runtime(config)
    rows, data = inspect_dataset(config, decode=True)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    write(root / "dataset_manifest.json", data)
    results = []
    for architecture in config["models"]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        path, init = initialization(architecture)
        model = build_model(architecture, path)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        sample = [r for r in rows if r["split"] == "0"][:config["batch_size"] * 3]
        scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
        for images, labels, _, _ in loader(config, sample, training=True, seed=42):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["amp"]):
                loss = torch.nn.functional.cross_entropy(model(images.cuda()), labels.cuda())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite capacity loss")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        results.append({**init, "batch_size": config["batch_size"], "optimizer_steps": 3,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "elapsed_seconds": time.monotonic() - start, "status": "PASS"})
        print(json.dumps(results[-1]), flush=True)
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    receipt = {"status": "PASS", "config_hash": identity(config), "data_sha256": data["data_sha256"],
               "source": source_identity(require_clean=False), "gpu": torch.cuda.get_device_name(), "models": results,
               "limitation": "Three optimizer steps per model verify capacity, not model convergence or scientific validity"}
    write(root / "capacity.json", receipt)
    return receipt


def train_one(config, architecture, seed, rows, output, run_identity):
    torch = torch_runtime(config)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["input_hash"] != run_identity or sha(root / "best.npz") != result["checkpoint_sha256"] or sha(root / "test_predictions.csv") != result["predictions_sha256"]:
            raise RuntimeError("Completed run identity/checkpoint/predictions changed")
        return result
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    weight_path, init = initialization(architecture)
    model = build_model(architecture, weight_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    train = [r for r in rows if r["split"] == "0"]
    validation = [r for r in rows if r["split"] == "1"]
    test = [r for r in rows if r["split"] == "2"]
    train_counts = {f"{y}:{g}": sum(r["y"] == str(y) and r["place"] == str(g) for r in train) for y in (0, 1) for g in (0, 1)}
    progress_path = root / "progress.json"
    state = {"input_hash": run_identity, "epoch": 0, "best_accuracy": -1.0, "best_epoch": None, "history": []}
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state["input_hash"] != run_identity:
            raise RuntimeError("Resume identity changed")
        load_state(root / state["last_reference"], state["last_sha256"], model, optimizer)
        if sha(root / state["best_reference"]) != state["best_sha256"]:
            raise RuntimeError("Best checkpoint changed")
        scaler.load_state_dict(state["scaler"])
    start = time.monotonic()
    last_epoch = state["epoch"] if state.get("early_stopped") else config["epochs"]
    for epoch in range(state["epoch"] + 1, last_epoch + 1):
        torch.manual_seed(seed + epoch); torch.cuda.manual_seed_all(seed + epoch)
        model.train(); loss_sum = 0.0; count = 0
        for images, labels, _, _ in loader(config, train, training=True, seed=seed + epoch):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["amp"]):
                loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; stopped without scientific output")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(labels); count += len(labels)
        labels, _, predictions, _, _ = predict(model, loader(config, validation, training=False, seed=seed))
        accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
        entry = {"epoch": epoch, "train_loss": loss_sum/count, "validation_accuracy": accuracy}
        state["history"].append(entry)
        if accuracy > state["best_accuracy"]:
            state.update(best_accuracy=accuracy, best_epoch=epoch)
            state["best_reference"] = f"best-epoch{epoch:03d}.npz"
            state["best_sha256"] = save_state(root / state["best_reference"], model)
        state["epoch"] = epoch
        state["early_stopped"] = (epoch >= config["minimum_epochs"] and epoch - state["best_epoch"] >= config["early_stopping_patience"])
        if epoch % config["checkpoint_interval"] == 0 or epoch == config["epochs"] or state["early_stopped"]:
            state["last_reference"] = f"last-epoch{epoch:03d}.npz"
            state["last_sha256"] = save_state(root / state["last_reference"], model, optimizer)
            state["scaler"] = scaler.state_dict()
            write(progress_path, state)
        print(json.dumps({"model": architecture, "seed": seed, **entry}), flush=True)
        if state["early_stopped"]:
            break
    load_state(root / state["best_reference"], state["best_sha256"], model)
    state["best_sha256"] = save_state(root / "best.npz", model)
    y, g, pred, probs, indices = predict(model, loader(config, test, training=False, seed=seed))
    predictions_path = root / "test_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["img_id", "img_filename", "y", "place", "split", "prediction", "probability_0", "probability_1"])
        for index, prediction, probability in zip(indices, pred, probs):
            row = test[index]
            writer.writerow([row[k] for k in ("img_id", "img_filename", "y", "place", "split")] + [prediction, *probability])
    result = {"status": "COMPLETED", "architecture": architecture, "seed": seed, "input_hash": run_identity,
              "initialization": init, "checkpoint_sha256": state["best_sha256"], "best_epoch": state["best_epoch"],
              "best_validation_accuracy": state["best_accuracy"], "epochs_completed": state["epoch"],
              "early_stopped": state.get("early_stopped", False),
              "predictions_sha256": sha(predictions_path), "training_elapsed_seconds_this_session": time.monotonic()-start,
              "metrics": metrics(y, g, pred, train_group_counts=train_counts, seed=seed,
                                 iterations=config["bootstrap_iterations"], confidence=config["confidence_level"])}
    write(result_path, result)
    return result


def run_matrix(config, output, acceptance):
    source = source_identity(require_clean=config["require_clean_git"])
    torch = torch_runtime(config)
    import torchvision
    rows, data = inspect_dataset(config)
    receipt = json.loads(Path(acceptance).read_text(encoding="utf-8"))
    if receipt["status"] != "PASS" or receipt["config_hash"] != identity(config) or receipt["data_sha256"] != data["data_sha256"]:
        raise RuntimeError("Capacity receipt does not match frozen data/config")
    initialized = {a: initialization(a)[1] for a in config["models"]}
    if any(initialized[r["architecture"]]["sha256"] != r["sha256"] for r in receipt["models"]):
        raise RuntimeError("Initialization changed after capacity acceptance")
    inputs = {"config": config, "source": source, "data_sha256": data["data_sha256"], "initializations": initialized,
              "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name()}}
    fingerprint = identity(inputs)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["input_hash"] != fingerprint:
        raise RuntimeError("Existing matrix identity changed; use a new directory")
    write(manifest_path, {"input_hash": fingerprint, **inputs})
    write(root / "environment_receipt.json", {
        "source_archive_sha256": archive_source(root / "source_snapshot.zip", source["source_tree_sha256"]),
        "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "config_hash": identity(config), "data_sha256": data["data_sha256"],
        "source_policy": config.get("source_policy"),
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    })
    write(root / "dataset_manifest.json", data)
    results = []
    for architecture in config["models"]:
        for seed in config["seeds"]:
            result = train_one(config, architecture, seed, rows, root / f"{architecture}-seed{seed}", fingerprint)
            results.append(result)
            write(root / "matrix_progress.json", {"completed": len(results), "total": 6, "results": results})
    summary = {}
    for architecture in config["models"]:
        selected = [r for r in results if r["architecture"] == architecture]
        summary[architecture] = {key: {"mean": float(np.mean([r["metrics"][key] for r in selected])),
                                      "std": float(np.std([r["metrics"][key] for r in selected], ddof=1))}
                                 for key in ("accuracy", "balanced_accuracy", "adjusted_average_accuracy", "worst_group_accuracy")}
    write(root / "paper_model_results.json", {"status": "COMPLETED", "input_hash": fingerprint,
          "completed_training_runs": 6, "results": results, "summary": summary,
          "limitations": ["This matrix measures task-model behavior; it does not replace ShortcutMiner E1-E5, evidence promotion or causal background intervention."]})
    lines = ["# Waterbirds image-model validation", "", "Six independent ERM training runs; official splits; validation-only checkpoint selection.", "",
             "| Model | Test accuracy mean ± std | Worst-group accuracy mean ± std |", "|---|---|---|"]
    for architecture, values in summary.items():
        a, w = values["accuracy"], values["worst_group_accuracy"]
        lines.append(f"| {architecture} | {a['mean']:.4f} ± {a['std']:.4f} | {w['mean']:.4f} ± {w['std']:.4f} |")
    lines.extend(["", "These measurements are not automatic shortcut confirmation or background-causality proof.",
                  "Full metrics, group sizes, intervals, initialization/checkpoint/data/source identities and predictions accompany this table."])
    (root / "paper_model_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
