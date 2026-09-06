# Local Docker acceptance (Phase 6)

Run at the repository root. Docker/Compose are the only required host tools.
The root Compose file is the only service definition; no Docker-in-Docker is used.

## Build, import, and tests

```bash
docker compose config
docker compose build
docker compose run --rm shortcutminer python -m shortcutminer --help
docker compose run --rm shortcutminer python -c "import shortcutminer"
docker compose run --rm shortcutminer pytest -q
```

Keep `SHORTCUTMINER_NETWORK_MODE=none` (the default) and do not pass API credentials
for acceptance commands. The build needs registry access, but the resulting mock
and Python tests execute without external network access.

## Result and checkpoint persistence across removed containers

Use fresh run IDs if these examples have already been executed. Never overwrite
an existing experiment to repeat an acceptance test.

```bash
docker compose run --rm shortcutminer python -m shortcutminer mock --run-id phase6-smoke --runs-root /runs
docker compose run --rm shortcutminer python scripts/container_acceptance.py phase6-smoke
docker compose down
docker compose run --rm shortcutminer python scripts/container_acceptance.py phase6-smoke
```

Both verification calls must return `verification: PASS`, matching hashes for
the durable JSON/event artifacts, and a readable SQLite checkpoint. Every
`run --rm` creates and removes a different container while keeping the named
volume. The verifier checks all five stage artifacts, run metadata, final report,
checkpoint thread ID, and SQLite integrity.

## Failed run and resume in a new container

The first command below deliberately exits **1**, after the first four agents
complete and Summarize fails. That status is expected only for this command.

```bash
docker compose run --rm shortcutminer python -m shortcutminer mock --run-id phase6-resume --runs-root /runs --fail-at summarize
docker compose run --rm shortcutminer python scripts/container_acceptance.py phase6-resume --failed
docker compose down
docker compose run --rm shortcutminer python -m shortcutminer resume phase6-resume --runs-root /runs --runtime mock
docker compose run --rm shortcutminer python scripts/container_acceptance.py phase6-resume --resumed
```

The resumed CLI output must list only `summarize` in `mock_agents_invoked`.
The verifier requires two attempts and exactly one successful event for each
earlier stage. Resuming a completed run again must invoke no agents.
Mock CLI resume is restricted to synthetic demo runs to avoid mixing canned
findings with real model experiments.

## Storage, permissions, and configuration

The named `runs` volume stores durable state below `/runs`. Its initial ownership
comes from the image's non-root UID/GID 10001. Do not use `docker compose down -v`
unless you intentionally want to delete these results. Custom bind mounts must
be writable by UID/GID 10001. Mount datasets read-only at `/data`.

For live model runs, follow the explicit environment and configuration commands
in the README. Keep the same `/data` and `/runs` paths when resuming so checkpoint
references remain valid. Change model YAML through `/config`, not a rebuild.
The image includes no raw datasets, host virtual environments, or `.env` files.
The CLI is a local batch process. No HTTP health endpoint, supervisor, external
database service, distributed runtime, or GPU stack is introduced.

The current repository scope remains local Docker execution. Real scientific
experiment backends and real-model integration require separate validation and
are not part of these container acceptance instructions.
