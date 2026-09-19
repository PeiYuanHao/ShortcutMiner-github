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

## Explicit real API mode

Real API execution uses the small `docker-compose.api.yml` override. Store credentials only in
the local, Git-ignored `.env.phase16` file:

```dotenv
SHORTCUTMINER_TEXT_API_KEY=
SHORTCUTMINER_VLM_API_KEY=
```

Do not place values in committed Compose/model files or pass them with `-e`. Check presence safely:

```powershell
docker compose -f docker-compose.yml -f docker-compose.api.yml run --rm shortcutminer python -m shortcutminer pilot credentials --config /config/experiments/real_blind_pilot.example.yaml
```

The API override enables bridge networking and loads `.env.phase16`. Without the explicit second
`-f`, the service remains offline and receives neither credential. The readiness command reports
only `SET` or `MISSING`.

Provider persistence uses a deny-by-schema boundary. The unambiguous readiness field is
`raw_provider_body_persistence=false`: raw HTTP response bodies, reasoning content,
Authorization headers, API keys, and image base64 are never written. Runtime snapshots retain
the legacy contract field `raw_response_persistence=false`; model-call artifacts contain only
sanitized identifiers, hashes, latency, usage, finish/schema status, and retry diagnostics.
The offline regression can be run without credentials or network access:

```bash
python scripts/phase16_provider_security_preflight.py --root /tmp/provider-security-preflight
```

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

The current repository scope remains local Docker execution. The bounded real CPU
quantitative backend and Pillow/NumPy qualitative descriptor backend are included in
the image and test suite. The qualitative backend can create task-scoped inspection
JSON and contact sheets in `/runs` while source datasets remain read-only under
`/data`. Provider-native VLM integration and large scientific evaluations remain
outside these container acceptance instructions.
