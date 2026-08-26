from collections.abc import Mapping
import json
import os
import tempfile
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
JsonObjectInput: TypeAlias = Mapping[str, JsonValue]


def atomic_write_json(
    path: Path,
    data: JsonObjectInput,
    *,
    error_type: type[Exception],
) -> None:
    """Serialize strict JSON and atomically replace ``path``."""

    temp_path: Path | None = None

    try:
        serialized = json.dumps(
            dict(data),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(serialized)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

        raise error_type(f"Failed to write artifact: {path}") from exc


def model_to_json_object(model: BaseModel) -> JsonObject:
    return model.model_dump(mode="json")
