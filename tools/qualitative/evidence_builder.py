from pathlib import Path
from typing import Any

from tools.qualitative.metadata_inspector import (
    normalize_dataset_reference,
    resolve_dataset_reference,
)


IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt"}


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return "text"
    return "unknown"


def _has_matching_image_signature(path: Path) -> bool:
    suffix = path.suffix.lower()
    with path.open("rb") as file:
        header = file.read(16)
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if suffix == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if suffix == ".bmp":
        return header.startswith(b"BM")
    if suffix in {".tif", ".tiff"}:
        return header.startswith((b"II*\x00", b"MM\x00*"))
    if suffix == ".webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    return False


def build_evidence_manifest(
    dataset_root: str,
    selected_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve selected records into immutable, traceable evidence entries."""

    manifest: list[dict[str, Any]] = []
    limitations: list[str] = []

    for record in selected_records:
        sample_id = record["sample_id"]
        raw_reference = record.get("sample_reference")
        entry = dict(record)
        entry["sample_available"] = False
        entry["media_type"] = None
        entry["byte_size"] = None

        if raw_reference is None or not str(raw_reference).strip():
            entry["sample_reference"] = None
            if record.get("metadata_only"):
                entry["sample_available"] = True
                entry["media_type"] = "tabular"
            else:
                limitations.append(f"Sample {sample_id} has no sample path in metadata.")
            manifest.append(entry)
            continue

        try:
            reference = normalize_dataset_reference(raw_reference)
            if reference is None:
                raise ValueError("empty sample path")
            path = resolve_dataset_reference(dataset_root, reference)
            entry["sample_reference"] = reference
            if not path.exists() or not path.is_file():
                limitations.append(f"Sample file is unavailable: {sample_id}.")
            else:
                entry["sample_available"] = True
                entry["media_type"] = _media_type(path)
                entry["byte_size"] = path.stat().st_size
                if entry["media_type"] == "unknown":
                    entry["sample_available"] = False
                    limitations.append(
                        f"Sample {sample_id} uses an unsupported or unknown format: "
                        f"{path.suffix or '<none>'}."
                    )
                elif (
                    entry["media_type"] == "image"
                    and not _has_matching_image_signature(path)
                ):
                    entry["sample_available"] = False
                    limitations.append(
                        f"Sample {sample_id} does not match its declared image format."
                    )
        except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
            entry["sample_reference"] = None
            limitations.append(f"Sample path was rejected for {sample_id}: {exc}")

        manifest.append(entry)

    return manifest, list(dict.fromkeys(limitations))
