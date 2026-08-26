from pathlib import Path, PurePosixPath, PureWindowsPath


class RunPathError(ValueError):
    """Raised when a run identifier cannot map to a safe run directory."""


def get_run_dir(
    runs_root: str | Path,
    run_id: str,
) -> Path:
    """
    Return a resolved ``runs_root/run_id`` directory path.

    A run ID is one opaque, portable path component using the same character
    policy as the protocol layer. Resolving both paths also prevents an
    existing symlink or junction from escaping ``runs_root``.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise RunPathError(
            "run_id must not be empty."
        )

    if len(run_id) > 128:
        raise RunPathError(
            "run_id must contain at most 128 characters."
        )

    if (
        PurePosixPath(run_id).is_absolute()
        or PureWindowsPath(run_id).is_absolute()
        or bool(PureWindowsPath(run_id).drive)
    ):
        raise RunPathError(
            "run_id must not be an absolute path."
        )

    if not run_id[0].isascii() or not run_id[0].isalnum():
        raise RunPathError(
            "run_id must begin with an ASCII letter or digit."
        )

    if not all(
        character.isascii()
        and (character.isalnum() or character in "._-")
        for character in run_id
    ):
        raise RunPathError(
            "run_id may contain only ASCII letters, digits, '.', '_', and '-'."
        )

    windows_stem = run_id.split(
        ".",
        maxsplit=1,
    )[0].upper()
    reserved_windows_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }

    if windows_stem in reserved_windows_names:
        raise RunPathError(
            "run_id must not be a reserved filesystem name."
        )

    if run_id.endswith("."):
        raise RunPathError(
            "run_id must not end with a dot."
        )

    try:
        root = Path(runs_root).resolve(
            strict=False
        )
        run_dir = (root / run_id).resolve(
            strict=False
        )

    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RunPathError(
            "Unable to resolve the run directory."
        ) from exc

    try:
        relative_run_dir = run_dir.relative_to(
            root
        )

    except ValueError as exc:
        raise RunPathError(
            "run directory must remain inside runs_root."
        ) from exc

    if relative_run_dir == Path("."):
        raise RunPathError(
            "run directory must be below runs_root."
        )

    return run_dir
