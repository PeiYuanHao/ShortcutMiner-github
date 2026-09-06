"""Durable DatasetProfile artifacts for run resume and traceability."""

import json
from pathlib import Path

from pydantic import ValidationError

from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import RunPathError, get_run_dir


class DatasetProfileStoreError(RuntimeError):
    pass


def dataset_profile_path(runs_root: str | Path, run_id: str) -> Path:
    try:
        return get_run_dir(runs_root, run_id) / "dataset" / "profile.json"
    except RunPathError as exc:
        raise DatasetProfileStoreError(str(exc)) from exc


def dataset_profile_exists(runs_root: str | Path, run_id: str) -> bool:
    return dataset_profile_path(runs_root, run_id).is_file()


def save_dataset_profile(
    runs_root: str | Path,
    run_id: str,
    profile: DatasetProfile,
) -> Path:
    path = dataset_profile_path(runs_root, run_id)
    if path.exists():
        existing = load_dataset_profile(runs_root, run_id)
        if existing != profile:
            raise DatasetProfileStoreError(
                "A different DatasetProfile already exists for this run."
            )
        return path
    atomic_write_json(
        path,
        profile.model_dump(mode="json"),
        error_type=DatasetProfileStoreError,
    )
    return path


def load_dataset_profile(
    runs_root: str | Path,
    run_id: str,
) -> DatasetProfile:
    path = dataset_profile_path(runs_root, run_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return DatasetProfile.model_validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise DatasetProfileStoreError(
            f"Unable to load a compatible DatasetProfile: {path}"
        ) from exc


__all__ = [
    "DatasetProfileStoreError",
    "dataset_profile_exists",
    "dataset_profile_path",
    "load_dataset_profile",
    "save_dataset_profile",
]
