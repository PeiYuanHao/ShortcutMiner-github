FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# The same dependency declaration is used for local and container tests.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Explicit allowlist: never copy the repository root or host credentials.
COPY pyproject.toml README.md ./
COPY shortcutminer/ ./shortcutminer/
COPY tools/ ./tools/
COPY agents/ ./agents/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
RUN python -m pip install --no-cache-dir --no-deps -e . \
    && groupadd --gid 10001 shortcutminer \
    && useradd --uid 10001 --gid shortcutminer --create-home shortcutminer \
    && mkdir -p /runs /data /app/.pytest_cache \
    && chown shortcutminer:shortcutminer /runs /data /app/.pytest_cache

USER shortcutminer
CMD ["python", "-m", "shortcutminer", "mock", "--runs-root", "/runs"]
