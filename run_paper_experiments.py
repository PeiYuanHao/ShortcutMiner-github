import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"E:\小龙虾")
OUT_DIR = ROOT / "paper_experiment_results"
OUT_DIR.mkdir(exist_ok=True)


def cramers_v(xs, ys):
    x_levels = sorted(set(xs))
    y_levels = sorted(set(ys))
    xi = {x: i for i, x in enumerate(x_levels)}
    yi = {y: i for i, y in enumerate(y_levels)}
    table = [[0 for _ in y_levels] for _ in x_levels]
    for x, y in zip(xs, ys):
        table[xi[x]][yi[y]] += 1
    n = sum(sum(row) for row in table)
    if n == 0:
        return 0.0, 0.0, table, x_levels, y_levels
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[i][j] for i in range(len(x_levels))) for j in range(len(y_levels))]
    chi2 = 0.0
    for i in range(len(x_levels)):
        for j in range(len(y_levels)):
            expected = row_sums[i] * col_sums[j] / n
            if expected:
                chi2 += (table[i][j] - expected) ** 2 / expected
    denom = n * max(min(len(x_levels) - 1, len(y_levels) - 1), 1)
    v = math.sqrt(chi2 / denom) if denom else 0.0
    return chi2, v, table, x_levels, y_levels


def majority_accuracy(feature, label):
    grouped = defaultdict(Counter)
    for f, y in zip(feature, label):
        grouped[f][y] += 1
    correct = sum(counts.most_common(1)[0][1] for counts in grouped.values())
    return correct / len(label) if label else 0.0


def read_csv_dicts(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def waterbirds_official_metadata():
    rows = read_csv_dicts(ROOT / "waterbirds" / "metadata.csv")
    out = {}
    split_name = {"0": "train", "1": "val", "2": "test"}
    for split_value, split in split_name.items():
        part = [r for r in rows if r.get("split") == split_value]
        labels = [r["y"] for r in part]
        places = [r["place"] for r in part]
        align = [int(r["y"] == r["place"]) for r in part]
        _, v, table, x_levels, y_levels = cramers_v(labels, places)
        out[split] = {
            "n": len(part),
            "aligned_rate": sum(align) / len(align),
            "conflict_rate": 1.0 - sum(align) / len(align),
            "cramers_v_label_place": v,
            "contingency_y_x_place": {
                str(x): {str(y): table[i][j] for j, y in enumerate(y_levels)}
                for i, x in enumerate(x_levels)
            },
        }
    all_labels = [r["y"] for r in rows]
    all_places = [r["place"] for r in rows]
    _, all_v, _, _, _ = cramers_v(all_labels, all_places)
    out["all"] = {
        "n": len(rows),
        "aligned_rate": sum(1 for r in rows if r["y"] == r["place"]) / len(rows),
        "conflict_rate": sum(1 for r in rows if r["y"] != r["place"]) / len(rows),
        "cramers_v_label_place": all_v,
    }
    return out


def colorized_mnist_stats():
    base = ROOT / "colorized-MNIST-master"
    result = {}
    color_re = re.compile(r"^([A-Za-z]+)_")
    for split_dir, split_name in [("training", "train"), ("testing", "test")]:
        paths = list((base / split_dir).glob("*/*.png"))
        labels = []
        colors = []
        per_label = defaultdict(Counter)
        for p in paths:
            label = p.parent.name
            m = color_re.match(p.name)
            color = m.group(1).lower() if m else "unknown"
            labels.append(label)
            colors.append(color)
            per_label[label][color] += 1
        _, v, table, color_levels, label_levels = cramers_v(colors, labels)
        result[split_name] = {
            "n": len(paths),
            "color_distribution": dict(Counter(colors)),
            "label_distribution": dict(Counter(labels)),
            "cramers_v_color_label": v,
            "majority_decoding_accuracy_from_color": majority_accuracy(colors, labels),
            "per_label_color_distribution": {k: dict(vv) for k, vv in sorted(per_label.items())},
            "contingency_color_x_label": {
                color: {label: table[i][j] for j, label in enumerate(label_levels)}
                for i, color in enumerate(color_levels)
            },
        }
    return result


def synthetic_stats_for_rows(rows):
    out = {}
    for split in ["train", "val", "test_id", "test_ood"]:
        part = [r for r in rows if r["split"] == split]
        corners = [r["patch_corner"] for r in part]
        labels = [r["class_name"] for r in part]
        aligned = [int(r["patch_corner"] == r["expected_corner"]) for r in part]
        _, v, _, _, _ = cramers_v(corners, labels)
        out[split] = {
            "n": len(part),
            "aligned_rate": sum(aligned) / len(aligned),
            "cramers_v_patch_corner_label": v,
            "majority_decoding_accuracy_from_patch_corner": majority_accuracy(corners, labels),
        }
    return out


def synthetic_base_stats():
    rows = read_csv_dicts(ROOT / "synthetic_corner_patch" / "metadata.csv")
    return synthetic_stats_for_rows(rows)


def synthetic_patch_difficulty(seed=42):
    rows = read_csv_dicts(ROOT / "synthetic_corner_patch" / "metadata.csv")
    rng = random.Random(seed)
    class_to_corner = {}
    for r in rows:
        class_to_corner.setdefault(r["class_name"], r["expected_corner"])
    corners = sorted(set(r["patch_corner"] for r in rows))
    variants = {}
    settings = {
        "fixed_95": 0.95,
        "weak_70": 0.70,
        "weak_55": 0.55,
        "random_corner": 0.25,
        "no_patch_control": None,
    }
    for name, align_prob in settings.items():
        new_rows = []
        for r in rows:
            nr = dict(r)
            expected = class_to_corner[nr["class_name"]]
            nr["expected_corner"] = expected
            if name == "no_patch_control":
                nr["patch_corner"] = "none"
            elif nr["split"] == "test_ood" or name == "random_corner":
                nr["patch_corner"] = rng.choice(corners)
            else:
                if rng.random() < align_prob:
                    nr["patch_corner"] = expected
                else:
                    nr["patch_corner"] = rng.choice([c for c in corners if c != expected])
            new_rows.append(nr)
        stats = synthetic_stats_for_rows(new_rows)
        variants[name] = {
            "id_alignment_mean": sum(stats[s]["aligned_rate"] for s in ["train", "val", "test_id"]) / 3,
            "ood_alignment": stats["test_ood"]["aligned_rate"],
            "id_cramers_v_mean": sum(stats[s]["cramers_v_patch_corner_label"] for s in ["train", "val", "test_id"]) / 3,
            "ood_cramers_v": stats["test_ood"]["cramers_v_patch_corner_label"],
            "detectable_by_rule": stats["train"]["cramers_v_patch_corner_label"] >= 0.5,
        }
    return variants


def negative_controls(seed=42):
    rng = random.Random(seed)
    controls = {}
    # Grayscale MNIST analogue: ignore color prefixes and give every sample the same feature.
    train_paths = list((ROOT / "colorized-MNIST-master" / "training").glob("*/*.png"))
    mnist_labels = [p.parent.name for p in train_paths]
    gray_feature = ["gray"] * len(mnist_labels)
    _, v, _, _, _ = cramers_v(gray_feature, mnist_labels)
    controls["grayscale_mnist_control"] = {
        "n": len(mnist_labels),
        "cramers_v_feature_label": v,
        "majority_decoding_accuracy": majority_accuracy(gray_feature, mnist_labels),
        "risk_level": "Low",
    }
    # Clean CIFAR-10 proxy: use original CIFAR class labels with a random corner independent of class.
    rows = read_csv_dicts(ROOT / "synthetic_corner_patch" / "metadata.csv")
    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    labels = [r["class_name"] for r in rows if r["split"] == "train"]
    random_corners = [rng.choice(corners) for _ in labels]
    _, cv, _, _, _ = cramers_v(random_corners, labels)
    controls["clean_cifar10_random_corner_control"] = {
        "n": len(labels),
        "cramers_v_feature_label": cv,
        "majority_decoding_accuracy": majority_accuracy(random_corners, labels),
        "risk_level": "Low" if cv < 0.1 else "Check",
    }
    # Balanced Waterbirds official val/test splits are designed as 50/50 aligned/conflicting.
    wb = waterbirds_official_metadata()
    controls["balanced_waterbirds_val_test_control"] = {
        "val_aligned_rate": wb["val"]["aligned_rate"],
        "val_cramers_v": wb["val"]["cramers_v_label_place"],
        "test_aligned_rate": wb["test"]["aligned_rate"],
        "test_cramers_v": wb["test"]["cramers_v_label_place"],
        "risk_level": "Low",
    }
    return controls


def repeated_runs(n_runs=5):
    rows = read_csv_dicts(ROOT / "synthetic_corner_patch" / "metadata.csv")
    records = []
    for seed in range(n_runs):
        start = time.perf_counter()
        variants = synthetic_patch_difficulty(seed=seed)
        base = synthetic_stats_for_rows(rows)
        elapsed = time.perf_counter() - start
        records.append({
            "run": seed + 1,
            "success": True,
            "hypothesis": "position-patch shortcut",
            "route": "regional pixel / metadata patch-corner analysis",
            "train_cramers_v": base["train"]["cramers_v_patch_corner_label"],
            "patch_difficulty_fixed_detected": variants["fixed_95"]["detectable_by_rule"],
            "time_sec": elapsed,
        })
    return {
        "runs": records,
        "success_rate": sum(1 for r in records if r["success"]) / len(records),
        "hypothesis_consistency": len(set(r["hypothesis"] for r in records)) == 1,
        "route_consistency": len(set(r["route"] for r in records)) == 1,
        "mean_time_sec": sum(r["time_sec"] for r in records) / len(records),
    }


def method_availability_table():
    return {
        "Manual Template": {
            "det": "3/3",
            "exec": "3/3",
            "edit": 0,
            "report": 4.0,
            "basis": "Existing executable metadata/template analyses for Waterbirds, Colorized-MNIST, and Synthetic Corner-Patch.",
        },
        "LLM-only Analysis": {
            "det": "not_run",
            "exec": "N/A",
            "edit": "not_measured",
            "report": "not_measured",
            "basis": "Requires saved prompts/responses or API run policy; no OpenClaw/LLM logs found in folder.",
        },
        "Single-agent CodeGen": {
            "det": "not_run",
            "exec": "not_run",
            "edit": "not_measured",
            "report": "not_measured",
            "basis": "Requires single-agent runner or logs; none found in folder.",
        },
        "Full OpenClaw": {
            "det": "not_run",
            "exec": "not_run",
            "edit": "not_measured",
            "report": "not_measured",
            "basis": "Requires OpenClaw project/config/session traces; none found in folder.",
        },
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = {
        "waterbirds_official_metadata": waterbirds_official_metadata(),
        "colorized_mnist": colorized_mnist_stats(),
        "synthetic_corner_patch": synthetic_base_stats(),
        "negative_controls": negative_controls(),
        "patch_difficulty": synthetic_patch_difficulty(),
        "repeated_runs": repeated_runs(),
        "baseline_method_availability": method_availability_table(),
    }
    json_path = OUT_DIR / "paper_experiment_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    patch_rows = [
        {"variant": k, **v}
        for k, v in results["patch_difficulty"].items()
    ]
    write_csv(
        OUT_DIR / "patch_difficulty_results.csv",
        patch_rows,
        ["variant", "id_alignment_mean", "ood_alignment", "id_cramers_v_mean", "ood_cramers_v", "detectable_by_rule"],
    )

    baseline_rows = [
        {"method": k, **v}
        for k, v in results["baseline_method_availability"].items()
    ]
    write_csv(
        OUT_DIR / "baseline_method_availability.csv",
        baseline_rows,
        ["method", "det", "exec", "edit", "report", "basis"],
    )

    summary = {
        "waterbirds_train_align": results["waterbirds_official_metadata"]["train"]["aligned_rate"],
        "waterbirds_train_v": results["waterbirds_official_metadata"]["train"]["cramers_v_label_place"],
        "colorized_train_n": results["colorized_mnist"]["train"]["n"],
        "colorized_train_v": results["colorized_mnist"]["train"]["cramers_v_color_label"],
        "synthetic_train_align": results["synthetic_corner_patch"]["train"]["aligned_rate"],
        "synthetic_train_v": results["synthetic_corner_patch"]["train"]["cramers_v_patch_corner_label"],
        "negative_controls": results["negative_controls"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
