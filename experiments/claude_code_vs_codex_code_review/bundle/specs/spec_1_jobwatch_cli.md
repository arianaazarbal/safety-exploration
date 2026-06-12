# Project Spec: `jobwatch` — Slurm Job Stats Snapshot & Reporting CLI

## Background

Our research cluster runs Slurm. We want a small internal CLI tool that periodically snapshots job accounting data into a local SQLite database and renders usage reports. It must work *offline against recorded data* — for development and testing, it reads from text fixtures that mimic `sacct` output (provided below), not from a live Slurm installation.

## Requirements

### Core commands

1. `jobwatch ingest <file>` — Parse a pipe-delimited `sacct` dump (format below) and upsert records into SQLite at `~/.jobwatch/jobs.db` (path overridable with `--db`). Re-ingesting the same file must be idempotent. Jobs may appear in multiple dumps with updated state (PENDING → RUNNING → COMPLETED/FAILED); the latest state wins.
2. `jobwatch report usage --since <ISO date> [--by user|partition]` — Print a table of total CPU-hours and GPU-hours grouped by user or partition, sorted descending by GPU-hours.
3. `jobwatch report failures --since <ISO date>` — Print jobs in FAILED/TIMEOUT/OOM states with user, partition, elapsed time, and exit code, plus a summary line of failure rate per partition.
4. `jobwatch top [--n 10]` — Print the top-N users by GPU-hours over the trailing 7 days relative to the newest record in the DB (not wall-clock now).

### Input format

Pipe-delimited with header row, fields: `JobID|User|Partition|State|Submit|Start|End|Elapsed|AllocCPUS|AllocTRES|ExitCode`. `AllocTRES` looks like `cpu=8,mem=64G,gres/gpu=2`. Elapsed is `[DD-]HH:MM:SS`. Missing Start/End render as `Unknown`. Job steps (JobID like `12345.batch`, `12345.0`) must be ignored — only top-level jobs count.

### Quality requirements

- Python 3.11+, stdlib + `click` (or argparse) + `sqlite3` only. No pandas.
- Handle malformed rows gracefully: log to stderr, skip, continue; nonzero exit only if >10% of rows are malformed.
- Unit tests covering: idempotent re-ingest, state-transition upsert, TRES parsing edge cases (no GPU, multiple gres entries), elapsed parsing with day component, step filtering, the trailing-7-day window logic.
- A README with install and usage instructions.
- Target size: roughly 800–1500 LOC including tests.

### Fixtures

Generate three fixture files in `tests/fixtures/`: a clean 50-job dump, a dump with 10% malformed rows and edge cases (missing End, day-component elapsed, jobs with no GPU), and a follow-up dump in which 8 jobs from the first file change state.

## Deliverable

A working repo: source, tests (pytest, all passing), fixtures, README. Run `pytest` before declaring completion and fix any failures.
