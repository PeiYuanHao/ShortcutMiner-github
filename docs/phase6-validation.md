> Historical phase record. Current deployment uses native Windows conda `shortcutminer`;
> follow `deployment.md` and `formal-experiment-environment.md` for current commands.

# Phase 6 local container acceptance

Date: 2026-09-04. Status: **PHASE 6: PASS**.
No cloud deployment or live model API call was performed.

## 1. Final status

PASS: real Docker builds, container tests, offline five-agent execution,
persistence, and cross-container resume have all been executed successfully.

## 2. Modified files in Phase 6

- `infrastructure/docker/Dockerfile`
- `.env.example`, `README.md`, `docs/deployment.md`
- `requirements.txt`, `pyproject.toml`
- `tests/test_pipeline_mock.py`: reuse the packaged synthetic responses while
  preserving concurrency and fan-in assertions.
- `tests/test_dataset_inspector.py`: explicitly control a test fixture's directory
  enumeration order, preserve its assertions, and add two unordered-input cases.
  Production scanning retains its strict budget-plus-one read bound.

Existing uncommitted Phase 1–5 work was preserved.

## 3. New files in Phase 6

- `.dockerignore`
- `docker-compose.yml` (relocated existing placeholder, not a second configuration)
- `shortcutminer/__main__.py`, `shortcutminer/cli.py`
- `shortcutminer/runtime/demo.py`
- `scripts/container_acceptance.py`
- `configs/experiments/container_run.example.yaml`
- `tests/test_cli.py`
- This validation report.

## 4. Removed or moved files

`infrastructure/docker/docker-compose.yml` was moved to the repository root.
There were no remaining retired-platform Docker files to delete in this phase.

## 5. Base image

Official `python:3.11-slim`, verified runtime Python **3.11.16**, Linux/amd64.
Resolved base digest:
`sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534`.
No Conda, CUDA, desktop environment, or additional agent framework was added.

## 6. Working directory

`/app`.

## 7. Runtime user

`shortcutminer`, UID/GID **10001**, verified inside the actual container.
New named-volume ownership is initialized from the image. Custom Linux bind
mounts must be writable by UID/GID 10001; startup does not change host ownership.

## 8. Compose services

Exactly one: `shortcutminer`. Default network mode is `none`, restart is disabled,
Linux capabilities are dropped, and new privileges are prohibited.

## 9. Volumes

- Named `runs` -> `/runs`; observed name: `shortcutminer-github_runs`.
- Host `./data` -> `/data`, read-only.
- Host `./configs` -> `/config`, read-only.

## 10. Storage persistence

Run metadata, manifest, events, stage artifacts, and final reports are under
`/runs/<run_id>/` on the named volume. Synthetic four-row input is stored below
`/runs/.mock-data/`. Real datasets are never copied into the image.

## 11. Checkpoint persistence

`/runs/<run_id>/checkpoints.sqlite`, using the existing Phase 4 SQLite saver.
Acceptance checked the stored thread ID and SQLite integrity from new containers.

## 12. Mock execution

From the repository root:

```bash
docker compose run --rm shortcutminer
```

Default CMD is `python -m shortcutminer mock --runs-root /runs`.
It invokes Runner -> Orchestrator -> LangGraph -> all five Agents and exits.
No key, host Python, or host Conda is required; container networking is disabled.
The canned evidence is explicitly synthetic and not a scientific result.

## 13. Real model configuration

Use CLI `run --config /config/experiments/container_run.example.yaml
--model-config /config/models/model_b.yaml`. Edit the mounted YAML and put the credential
in the local Git-ignored `.env.phase16` file under the environment-variable name selected by
model `api_key_env`. Use the explicit `docker-compose.api.yml` override, which loads that file
and enables bridge networking. Do not pass credentials through `-e` or command history.
No rebuild is needed.
Resume uses `--runtime openai-compatible --model-config ...` with the same mounts.
Mounted request/model parsing and client configuration were verified without API
invocation. CLI unit tests also verify configuration forwarding with a fake client.

## 14. Build result

`docker compose build`: **exit 0**, `Image shortcutminer-github-shortcutminer Built`.
A subsequent repeat build also exited 0 with all application layers cached.
Final image ID:
`sha256:ff327ac8c770b9a03af79ca2258eeb35f993c9a27f50d964a8dbf989b244049a`.

Earlier attempts hit Docker Hub connectivity, an incomplete-download hash
mismatch, and resolver backtracking. None were bypassed. The shared requirements
now pin verified top-level versions: LangGraph 1.2.11, SQLite saver 3.1.1,
Pydantic 2.13.5, Pandas 3.0.5, NumPy 2.4.6, PyYAML 6.0.3, pytest 9.1.1.
Package compatibility ranges remain in pyproject.toml. This is not a full
transitive lockfile or a claim of byte-for-byte reproducible future builds.

## 15. Container pytest result

`docker compose run --rm shortcutminer pytest -q`: **exit 0**.

```text
285 passed in 2.83s
```

No skip or xfail was added. Linux runs the three symlink tests that the Windows
host cannot run. Host conda regression: `282 passed, 3 skipped in 4.56s`.
Container `python -m pip check`: `No broken requirements found.`

## 16. Mock smoke result

Default container command: **exit 0**, `COMPLETED`.
Run: `run-20260904T140300416243Z-88f467c16369`.
Invoked: Identify, Executive, Quantitative, Qualitative, Summarize.
Final artifact: `/runs/run-20260904T140300416243Z-88f467c16369/final_result.json`.
CLI help and package import checks also passed.

## 17. Persistence result

PASS. Containers were removed with `--rm`, Compose was stopped without deleting
volumes, the image was rebuilt, and fresh containers verified the prior run.
All eleven JSON/event artifact hashes remained identical. Final-result SHA256:
`d6ee4809db68c7a7c32f3c620790f56ddabb4546babdd5924695c91c8eec69c1`.
The persisted checkpoint remained readable and passed SQLite integrity checking.

## 18. Resume result

PASS. Run `phase6-resume-20260904` deliberately failed at Summarize (expected exit
1). A new container verified FAILED status and all four earlier stage artifacts.
Another container resumed the same run/volume and exited 0, invoking **only
Summarize**. Verification showed two attempts, one completion per earlier stage,
and a final report. Resuming the completed run again invoked **zero** agents.
This validates failure recovery across removed containers, not a mid-request
SIGTERM fault-injection experiment. No new process manager or signal wrapper exists.

## 19. Retired-platform dependency audit

Case-insensitive searches for the retired runtime/client/session markers in Docker,
Compose, scripts, and environment files returned **zero matches**. Package inspection
inside the image also found no retired runtime installation.

## 20. Secret and image checks

PASS for the inspected image/configuration: no `.env` files, real API-key environment
values, copied datasets, or previous run outputs. Docker uses explicit COPY paths
plus `.dockerignore` exclusions for credentials, virtual environments, datasets,
outputs, and caches. Keys are not embedded in Dockerfile/build history. The bare
image had empty `/data` and `/runs`, UID 10001, and only the loopback interface when
started with `--network none`. No live API call was made during acceptance.

## 21. Linux CPU container readiness

The local Linux container foundation is validated and does not require host
Python/Conda or GPU support. Logs are emitted on stdout/stderr, with durable run
events on the volume.

## 22. Historical Phase 6 boundary

This record predates the completed Phase 7 Tool Runtime work. At the end of Phase 6,
live API validation, real experiment backends, operational hardening, and real
experiments remained future work. No external infrastructure, public endpoints,
GPU images, or large-scale experiments were created. Acceptance run data was
retained in the named volume; do not use `docker compose down -v` unless deletion
is intended.
