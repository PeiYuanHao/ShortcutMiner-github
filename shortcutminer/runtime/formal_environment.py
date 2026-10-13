"""Load local provider secrets only for the lifetime of a live CLI command."""

from contextlib import contextmanager
import os
from pathlib import Path
import re

FORMAL_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.formal"
CREDENTIAL_NAMES = ("SHORTCUTMINER_TEXT_API_KEY", "SHORTCUTMINER_VLM_API_KEY")


def _read_credentials(path):
    if not path.is_file():
        return {}
    credentials = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if match is None:
            raise ValueError(f"Invalid .env.formal assignment at line {number}")
        name, value = match.groups()
        if name not in CREDENTIAL_NAMES:
            continue
        if name in credentials:
            raise ValueError(f"Duplicate .env.formal assignment at line {number}")
        if value.startswith(("'", '"')):
            quote = value[0]
            match = re.fullmatch(re.escape(quote) + r"(.*?)" + re.escape(quote) + r"\s*(?:#.*)?", value)
            if match is None:
                raise ValueError(f"Invalid .env.formal quoting at line {number}")
            value = match.group(1)
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        credentials[name] = value
    return credentials


@contextmanager
def formal_credentials(*, enabled=True, env_file=None):
    """Existing explicit credentials take precedence; restore all injected values."""
    if not enabled or os.environ.get("SHORTCUTMINER_DISABLE_FORMAL_ENV") == "1":
        yield
        return
    credentials = _read_credentials(Path(env_file) if env_file is not None else FORMAL_ENV_FILE)
    injected = []
    try:
        for name, value in credentials.items():
            if value and not os.environ.get(name):
                previous = os.environ.get(name)
                os.environ[name] = value
                injected.append((name, previous))
        yield
    finally:
        for name, previous in injected:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def command_uses_formal_credentials(args):
    if args.command in {"audit", "run"}:
        return True
    if args.command == "resume":
        return args.runtime == "openai-compatible"
    return args.command == "pilot" and args.pilot_command in {
        "credentials", "preflight", "identify-diagnostic", "qualitative-diagnostic",
        "quantitative-diagnostic", "canary", "run", "resume",
    }
