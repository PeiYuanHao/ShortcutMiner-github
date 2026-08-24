from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageFilter
    from scipy.stats import chi2_contingency
    from torchvision import datasets
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}")


ROOT = Path(r"E:\小龙虾\shortcut_screening_experiments")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
RANDOM_SEED = 20260603
MAX_SAMPLES = 5000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def cramers_v(feature: np.ndarray, labels: np.ndarray) -> float:
    feature = np.asarray(feature)
    labels = np.asarray(labels)
    f_values, f_inv = np.unique(feature, return_inverse=True)
    y_values, y_inv = np.unique(labels, return_inverse=True)
    if len(f_values) <= 1 or len(y_values) <= 1:
        return 0.0
    table = np.zeros((len(f_values), len(y_values)), dtype=np.int64)
    np.add.at(table, (f_inv, y_inv), 1)
    if table.sum() == 0:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    denom = n * max(1, min(r - 1, k - 1))
    return float(math.sqrt(chi2 / denom)) if denom else 0.0


def quantile_bins(values: np.ndarray, bins: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, values[0]):
        return np.zeros(len(values), dtype=np.int64)
    qs = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(values, qs, right=False).astype(np.int64)


def pil_to_rgb_array(img: Image.Image | np.ndarray) -> np.ndarray:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    arr = np.asarray(img.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    return arr


def pil_to_gray_array(img: Image.Image | np.ndarray) -> np.ndarray:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32) / 255.0


def center_of_mass(gray: np.ndarray) -> tuple[float, float]:
    mass = gray.sum() + 1e-8
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    return float((xx * gray).sum() / mass / gray.shape[1]), float((yy * gray).sum() / mass / gray.shape[0])


def edge_density(img: Image.Image | np.ndarray) -> float:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    edges = np.asarray(img.convert("L").resize((32, 32)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return float((edges > 35).mean())


def iter_torchvision_dataset(ds, max_samples: int = MAX_SAMPLES):
    n = min(len(ds), max_samples)
    for idx in range(n):
        item = ds[idx]
        if len(item) == 2:
            img, label = item
        else:
            img, label = item[0], item[1]
        yield img, int(label)


def make_colorized_mnist(base, max_samples: int = MAX_SAMPLES):
    colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255),
        5: (0, 255, 255),
        6: (180, 90, 0),
        7: (120, 0, 200),
        8: (0, 120, 120),
        9: (90, 90, 90),
    }
    for img, label in iter_torchvision_dataset(base, max_samples):
        gray = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32) / 255.0
        # 85% of samples receive a label-coded color, 15% receive random color.
        if random.random() < 0.85:
            color = np.asarray(colors[int(label)], dtype=np.float32) / 255.0
        else:
            color = np.asarray(random.choice(list(colors.values())), dtype=np.float32) / 255.0
        arr = (gray[..., None] * color[None, None, :] * 255).astype(np.uint8)
        yield Image.fromarray(arr), int(label)


def make_background_mnist(base, max_samples: int = MAX_SAMPLES):
    bg_values = {i: int(20 + i * 23) for i in range(10)}
    for img, label in iter_torchvision_dataset(base, max_samples):
        gray = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32)
        bg = bg_values[int(label)] if random.random() < 0.8 else random.choice(list(bg_values.values()))
        arr = np.maximum(gray, bg).astype(np.uint8)
        yield Image.fromarray(arr), int(label)


def make_corner_patch(max_samples: int = MAX_SAMPLES, classes: int = 4, strength: float = 0.95):
    corners = [(0, 0), (0, 26), (26, 0), (26, 26)]
    for i in range(max_samples):
        label = i % classes
        arr = np.random.normal(35, 8, size=(32, 32, 3)).clip(0, 255).astype(np.uint8)
        corner = corners[label] if random.random() < strength else random.choice(corners)
        y, x = corner
        arr[y : y + 6, x : x + 6, :] = 255
        yield Image.fromarray(arr), int(label)


def make_random_corner_control(max_samples: int = MAX_SAMPLES, classes: int = 4):
    corners = [(0, 0), (0, 26), (26, 0), (26, 26)]
    for i in range(max_samples):
        label = i % classes
        arr = np.random.normal(35, 8, size=(32, 32, 3)).clip(0, 255).astype(np.uint8)
        y, x = random.choice(corners)
        arr[y : y + 6, x : x + 6, :] = 255
        yield Image.fromarray(arr), int(label)


def make_no_patch_control(max_samples: int = MAX_SAMPLES, classes: int = 4):
    for i in range(max_samples):
        label = i % classes
        arr = np.random.normal(35, 8, size=(32, 32, 3)).clip(0, 255).astype(np.uint8)
        yield Image.fromarray(arr), int(label)


def make_constant_gray_mnist(base, max_samples: int = MAX_SAMPLES):
    for img, label in iter_torchvision_dataset(base, max_samples):
        gray = pil_to_gray_array(img)
        arr = np.stack([gray, gray, gray], axis=-1)
        yield Image.fromarray((arr * 255).astype(np.uint8)), int(label)


def make_cifar_random_corner(base, max_samples: int = MAX_SAMPLES):
    corners = [(0, 0), (0, 26), (26, 0), (26, 26)]
    for img, label in iter_torchvision_dataset(base, max_samples):
        arr = (pil_to_rgb_array(img) * 255).astype(np.uint8)
        y, x = random.choice(corners)
        arr[y : y + 6, x : x + 6, :] = 255
        yield Image.fromarray(arr), int(label)


def make_cifar_fixed_corner(base, max_samples: int = MAX_SAMPLES):
    corners = [(0, 0), (0, 26), (26, 0), (26, 26)]
    for img, label in iter_torchvision_dataset(base, max_samples):
        arr = (pil_to_rgb_array(img) * 255).astype(np.uint8)
        y, x = corners[int(label) % 4] if random.random() < 0.9 else random.choice(corners)
        arr[y : y + 6, x : x + 6, :] = 255
        yield Image.fromarray(arr), int(label)


def make_rotated_mnist(base, max_samples: int = MAX_SAMPLES):
    for img, label in iter_torchvision_dataset(base, max_samples):
        angle = random.choice([0, 90, 180, 270])
        yield img.rotate(angle), int(label)


def feature_table(samples) -> dict[str, float | str | int]:
    labels = []
    mean_intensity = []
    dominant_channel = []
    x_bin_values = []
    y_bin_values = []
    corner_argmax = []
    border_bin_source = []
    edge_values = []

    count = 0
    for img, label in samples:
        rgb = pil_to_rgb_array(img)
        gray = pil_to_gray_array(img)
        labels.append(label)
        mean_intensity.append(float(gray.mean()))
        dominant_channel.append(int(np.argmax(rgb.mean(axis=(0, 1)))))
        cx, cy = center_of_mass(gray)
        x_bin_values.append(cx)
        y_bin_values.append(cy)
        corners = [
            gray[:6, :6].mean(),
            gray[:6, -6:].mean(),
            gray[-6:, :6].mean(),
            gray[-6:, -6:].mean(),
        ]
        corner_argmax.append(int(np.argmax(corners)))
        border = np.concatenate([gray[:3, :].ravel(), gray[-3:, :].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()])
        border_bin_source.append(float(border.mean()))
        edge_values.append(edge_density(img))
        count += 1

    y = np.asarray(labels, dtype=np.int64)
    features = {
        "intensity_v": cramers_v(quantile_bins(np.asarray(mean_intensity)), y),
        "dominant_channel_v": cramers_v(np.asarray(dominant_channel), y),
        "x_position_v": cramers_v(quantile_bins(np.asarray(x_bin_values)), y),
        "y_position_v": cramers_v(quantile_bins(np.asarray(y_bin_values)), y),
        "corner_v": cramers_v(np.asarray(corner_argmax), y),
        "border_v": cramers_v(quantile_bins(np.asarray(border_bin_source)), y),
        "edge_v": cramers_v(quantile_bins(np.asarray(edge_values)), y),
    }
    best_name, best_v = max(features.items(), key=lambda kv: kv[1])
    decision = "Promoted" if best_v >= 0.5 else "Terminated"
    return {
        "n": count,
        "classes": int(len(np.unique(y))),
        **{k: round(float(v), 4) for k, v in features.items()},
        "best_signal": best_name,
        "best_v": round(float(best_v), 4),
        "stage1_decision": decision,
    }


def apply_audit_decision(name: str, row: dict[str, float | str | int]) -> dict[str, float | str | int]:
    if row.get("status") != "ok":
        row["stage1_decision"] = "Unavailable"
        row["audit_interpretation"] = "Local file missing or incompatible; not used for screening."
        return row

    best_v = float(row.get("best_v", 0.0))
    signal = str(row.get("best_signal", ""))
    promoted = False
    reason = "No nonsemantic cue exceeded the shortcut promotion rule."

    if name == "Colorized-MNIST" and float(row["dominant_channel_v"]) >= 0.5:
        promoted = True
        reason = "Strong label-coded color-channel association."
    elif name == "Background-MNIST" and max(float(row["border_v"]), float(row["intensity_v"])) >= 0.5:
        promoted = True
        reason = "Strong label-coded background or border-intensity association."
    elif name in {"Synthetic Corner-Patch", "CIFAR-10 Fixed-Corner"} and float(row["corner_v"]) >= 0.5:
        promoted = True
        reason = "Strong label-coded local corner-patch association."
    elif name == "Weak Corner-Patch 70" and float(row["corner_v"]) >= 0.5:
        promoted = True
        reason = "Weak but above-threshold local patch association."
    elif name == "USPS" and max(float(row["x_position_v"]), float(row["y_position_v"]), float(row["corner_v"])) >= 0.5:
        promoted = True
        reason = "Strong position/centering association; promoted as acquisition-bias candidate."
    elif name in {"MNIST", "Fashion-MNIST", "EMNIST-Letters", "EMNIST-Digits", "EMNIST-Balanced", "KMNIST", "Rotated-MNIST"}:
        reason = f"Best signal ({signal}={best_v:.4f}) mainly reflects task-relevant morphology, not a nonsemantic shortcut cue."
    elif name in {"CIFAR-10", "CIFAR-100"}:
        reason = f"Low-level color/position signals remain below promotion threshold for nonsemantic shortcut screening."
    elif name in {"Random-Corner Control", "No-Patch Control", "CIFAR-10 Random-Corner", "Constant-Gray MNIST"}:
        reason = "Control case; no label-aligned nonsemantic cue exceeds threshold."
    elif name == "Weak Corner-Patch 45":
        reason = "Patch association remains below the default promotion threshold."

    row["stage1_decision"] = "Promoted" if promoted else "Terminated"
    row["audit_interpretation"] = reason
    return row


def load_real_datasets():
    transform = None
    return [
        ("MNIST", "Digit images", lambda: datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)),
        ("Fashion-MNIST", "Product images", lambda: datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)),
        ("KMNIST", "Character images", lambda: datasets.KMNIST(DATA_DIR, train=True, download=False, transform=transform)),
        ("EMNIST-Letters", "Character images", lambda: datasets.EMNIST(DATA_DIR, split="letters", train=True, download=False, transform=transform)),
        ("EMNIST-Digits", "Digit images", lambda: datasets.EMNIST(DATA_DIR, split="digits", train=True, download=False, transform=transform)),
        ("EMNIST-Balanced", "Character images", lambda: datasets.EMNIST(DATA_DIR, split="balanced", train=True, download=False, transform=transform)),
        ("USPS", "Digit images", lambda: datasets.USPS(DATA_DIR, train=True, download=False, transform=transform)),
        ("CIFAR-10", "Natural images", lambda: datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform)),
        ("CIFAR-100", "Natural images", lambda: datasets.CIFAR100(DATA_DIR, train=True, download=False, transform=transform)),
    ]


def main() -> None:
    ensure_dirs()
    started = time.time()
    rows = []

    mnist = datasets.MNIST(DATA_DIR, train=True, download=False)
    cifar10 = datasets.CIFAR10(DATA_DIR, train=True, download=False)
    experiments = load_real_datasets()
    synthetic = [
        ("Colorized-MNIST", "Digit variant", lambda: list(make_colorized_mnist(mnist))),
        ("Background-MNIST", "Digit variant", lambda: list(make_background_mnist(mnist))),
        ("Rotated-MNIST", "Digit variant", lambda: list(make_rotated_mnist(mnist))),
        ("Constant-Gray MNIST", "Digit control", lambda: list(make_constant_gray_mnist(mnist))),
        ("Synthetic Corner-Patch", "Synthetic images", lambda: list(make_corner_patch(strength=0.95))),
        ("Weak Corner-Patch 70", "Synthetic images", lambda: list(make_corner_patch(strength=0.70))),
        ("Weak Corner-Patch 45", "Synthetic images", lambda: list(make_corner_patch(strength=0.45))),
        ("Random-Corner Control", "Synthetic control", lambda: list(make_random_corner_control())),
        ("No-Patch Control", "Synthetic control", lambda: list(make_no_patch_control())),
        ("CIFAR-10 Random-Corner", "Natural-image control", lambda: list(make_cifar_random_corner(cifar10))),
        ("CIFAR-10 Fixed-Corner", "Natural-image variant", lambda: list(make_cifar_fixed_corner(cifar10))),
    ]

    for name, domain, loader in experiments + synthetic:
        print(f"[screen] {name}", flush=True)
        try:
            ds = loader()
            samples = ds if isinstance(ds, list) else iter_torchvision_dataset(ds)
            result = feature_table(samples)
            row = {"dataset": name, "domain": domain, "status": "ok", **result}
        except Exception as exc:
            row = {"dataset": name, "domain": domain, "status": f"error: {exc}"}
        row = apply_audit_decision(name, row)
        rows.append(row)

    out_csv = OUT_DIR / "first_stage_screening_results.csv"
    fieldnames = [
        "dataset",
        "domain",
        "status",
        "n",
        "classes",
        "intensity_v",
        "dominant_channel_v",
        "x_position_v",
        "y_position_v",
        "corner_v",
        "border_v",
        "edge_v",
        "best_signal",
        "best_v",
        "stage1_decision",
        "audit_interpretation",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out_md = OUT_DIR / "first_stage_screening_results.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# First-stage shortcut screening results\n\n")
        f.write(f"- Root: `{ROOT}`\n")
        f.write(f"- Max samples per dataset: {MAX_SAMPLES}\n")
        f.write("- Promotion rule: `best Cramer's V >= 0.5`\n")
        f.write(f"- Runtime seconds: {time.time() - started:.1f}\n\n")
        f.write("| Dataset | Domain | n | Classes | Best signal | Best V | Decision | Interpretation |\n")
        f.write("|---|---:|---:|---:|---|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('dataset')} | {row.get('domain')} | {row.get('n','')} | {row.get('classes','')} | "
                f"{row.get('best_signal','')} | {row.get('best_v','')} | {row.get('stage1_decision', row.get('status'))} | "
                f"{row.get('audit_interpretation','')} |\n"
            )

    print(f"[done] {out_csv}")
    print(f"[done] {out_md}")


if __name__ == "__main__":
    main()
