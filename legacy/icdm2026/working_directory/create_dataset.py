import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10


# ============================================================
# Synthetic Corner-Patch Dataset Builder
# ============================================================
# Purpose:
# Build a synthetic CIFAR-10 based shortcut dataset.
# A small red patch is inserted into image corners.
# In train / val / test_id, patch location is highly correlated
# with class label.
# In test_ood, patch location is random and independent of label.
# ============================================================


# ----------------------
# Basic settings
# ----------------------
DATA_ROOT = "./data"
OUT_ROOT = Path("./synthetic_corner_patch")

# Whether to delete old generated dataset before rebuilding
CLEAR_OLD_OUTPUT = True

# Use four CIFAR-10 classes for a clean 4-corner shortcut setting
# CIFAR-10 class index:
# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
SELECTED_CLASSES = [0, 1, 2, 3]

CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]

CLASS_TO_CORNER = {
    0: "top_left",       # airplane
    1: "top_right",      # automobile
    2: "bottom_left",    # bird
    3: "bottom_right",   # cat
}

PATCH_SIZE = 4
PATCH_COLOR = (255, 0, 0)  # same red patch for all classes

ALIGN_PROB = 0.95          # shortcut strength in train / val / test_id
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def add_corner_patch(img, corner, patch_size=4, color=(255, 0, 0)):
    """
    Add a small colored patch to one corner of a PIL image.

    Args:
        img: PIL.Image
        corner: one of top_left, top_right, bottom_left, bottom_right
        patch_size: patch size in pixels
        color: RGB color tuple

    Returns:
        PIL.Image with inserted patch
    """
    arr = np.array(img).copy()
    h, w, _ = arr.shape
    s = patch_size

    if corner == "top_left":
        arr[0:s, 0:s, :] = color
    elif corner == "top_right":
        arr[0:s, w - s:w, :] = color
    elif corner == "bottom_left":
        arr[h - s:h, 0:s, :] = color
    elif corner == "bottom_right":
        arr[h - s:h, w - s:w, :] = color
    else:
        raise ValueError(f"Unknown corner: {corner}")

    return Image.fromarray(arr)


def choose_corner(label, split):
    """
    Choose patch corner according to dataset split.

    train / val / test_id:
        patch corner is highly correlated with class label.

    test_ood:
        patch corner is randomly assigned, breaking the shortcut.
    """
    expected_corner = CLASS_TO_CORNER[label]

    if split in ["train", "val", "test_id"]:
        # 95% aligned with class-specific corner, 5% noise
        if random.random() < ALIGN_PROB:
            patch_corner = expected_corner
        else:
            other_corners = [c for c in CORNERS if c != expected_corner]
            patch_corner = random.choice(other_corners)

    elif split == "test_ood":
        # random corner, independent of class label
        patch_corner = random.choice(CORNERS)

    else:
        raise ValueError(f"Unknown split: {split}")

    is_aligned = int(patch_corner == expected_corner)
    return patch_corner, expected_corner, is_aligned


def collect_indices_by_class(dataset):
    """
    Collect sample indices for selected CIFAR-10 classes.
    Returns a dictionary:
        class_label -> list of indices
    """
    indices_by_class = {label: [] for label in SELECTED_CLASSES}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in SELECTED_CLASSES:
            indices_by_class[label].append(idx)

    return indices_by_class


def split_train_val_by_class(indices_by_class, val_ratio=0.2):
    """
    Split CIFAR-10 training set into train and val by class,
    keeping class balance approximately stable.
    """
    train_indices = []
    val_indices = []

    for label, indices in indices_by_class.items():
        indices = indices.copy()
        random.shuffle(indices)

        n_val = int(len(indices) * val_ratio)

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return train_indices, val_indices


def split_test_id_ood_by_class(indices_by_class):
    """
    Split CIFAR-10 test set into test_id and test_ood by class,
    keeping class balance approximately stable.
    """
    test_id_indices = []
    test_ood_indices = []

    for label, indices in indices_by_class.items():
        indices = indices.copy()
        random.shuffle(indices)

        half = len(indices) // 2

        test_id_indices.extend(indices[:half])
        test_ood_indices.extend(indices[half:])

    random.shuffle(test_id_indices)
    random.shuffle(test_ood_indices)

    return test_id_indices, test_ood_indices


def save_split(dataset, indices, split, writer):
    """
    Save one dataset split and write metadata records.
    """
    for new_id, original_index in enumerate(indices):
        img, label = dataset[original_index]
        class_name = dataset.classes[label]

        patch_corner, expected_corner, is_aligned = choose_corner(label, split)

        patched_img = add_corner_patch(
            img=img,
            corner=patch_corner,
            patch_size=PATCH_SIZE,
            color=PATCH_COLOR,
        )

        save_dir = OUT_ROOT / split / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{split}_{new_id:06d}.png"
        save_path = save_dir / filename

        patched_img.save(save_path)

        writer.writerow({
            "split": split,
            "image_path": str(save_path),
            "relative_path": str(save_path.relative_to(OUT_ROOT)),
            "original_index": original_index,
            "label": label,
            "class_name": class_name,
            "patch_corner": patch_corner,
            "expected_corner": expected_corner,
            "is_aligned": is_aligned,
            "patch_size": PATCH_SIZE,
            "patch_color": str(PATCH_COLOR),
            "align_prob": ALIGN_PROB if split in ["train", "val", "test_id"] else "random",
        })


def main():
    if CLEAR_OLD_OUTPUT and OUT_ROOT.exists():
        print(f"Removing old dataset folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10...")
    train_set = CIFAR10(root=DATA_ROOT, train=True, download=True)
    test_set = CIFAR10(root=DATA_ROOT, train=False, download=True)

    print("Collecting selected class indices...")
    train_indices_by_class = collect_indices_by_class(train_set)
    test_indices_by_class = collect_indices_by_class(test_set)

    print("Splitting train / val...")
    train_indices, val_indices = split_train_val_by_class(
        train_indices_by_class,
        val_ratio=VAL_RATIO
    )

    print("Splitting test_id / test_ood...")
    test_id_indices, test_ood_indices = split_test_id_ood_by_class(
        test_indices_by_class
    )

    metadata_path = OUT_ROOT / "metadata.csv"

    fieldnames = [
        "split",
        "image_path",
        "relative_path",
        "original_index",
        "label",
        "class_name",
        "patch_corner",
        "expected_corner",
        "is_aligned",
        "patch_size",
        "patch_color",
        "align_prob",
    ]

    print("Saving patched images and metadata...")

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        save_split(train_set, train_indices, "train", writer)
        save_split(train_set, val_indices, "val", writer)
        save_split(test_set, test_id_indices, "test_id", writer)
        save_split(test_set, test_ood_indices, "test_ood", writer)

    print("\nSynthetic Corner-Patch dataset has been created successfully.")
    print(f"Output folder: {OUT_ROOT.resolve()}")
    print(f"Metadata file: {metadata_path.resolve()}")

    print("\nDataset split sizes:")
    print(f"train:    {len(train_indices)}")
    print(f"val:      {len(val_indices)}")
    print(f"test_id:  {len(test_id_indices)}")
    print(f"test_ood: {len(test_ood_indices)}")


if __name__ == "__main__":
    main()