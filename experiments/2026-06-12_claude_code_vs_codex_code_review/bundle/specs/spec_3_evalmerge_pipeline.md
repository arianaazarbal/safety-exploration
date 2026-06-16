# Project Spec: `evalmerge` — Eval Results Normalization Pipeline

## Background

We receive model-evaluation results from three different harnesses, each in its own messy format. We need a pipeline that ingests all three, normalizes to a canonical schema, deduplicates, validates, loads to SQLite, and emits a data-quality report. Test data generation is part of the task (formats specified exactly below).

## Input formats (the pipeline must handle all three)

### Format A — `harness_a/*.csv`
Columns: `run_id,model,task,score,timestamp,meta`. Quirks the pipeline must survive: `score` is sometimes a percentage string ("87.5%") and sometimes a 0–1 float; `timestamp` is mixed (`2026-05-01T12:00:00Z` and `05/01/2026 12:00`); `meta` is a JSON string that is occasionally truncated/invalid; some files have a UTF-8 BOM; some rows have too many commas because `meta` was not quoted.

### Format B — `harness_b/*.jsonl`
One JSON object per line: `{"id", "model_name", "benchmark", "metrics": {"acc": float, "stderr": float}, "ts": unix_epoch_seconds, "tags": [str]}`. Quirks: occasional blank lines and lines of literal `null`; `ts` sometimes in milliseconds; `model_name` has inconsistent casing/spacing ("Gemma-2 2B" vs "gemma-2-2b"); duplicate `id`s across files where the later file's row is a correction.

### Format C — `harness_c/results.xml`
`<results><run id="..."><model>...</model><task>...</task><score unit="percent|fraction">...</score><date>YYYY-MM-DD</date></run>...</results>`. Quirks: a handful of runs are missing `<score>` entirely; dates have no time component (treat as midnight UTC); a few `unit` attributes are misspelled ("precent").

## Canonical schema (SQLite table `results`)

`run_id TEXT, source TEXT (a|b|c), model TEXT (normalized: lowercase, hyphenated), task TEXT, score REAL (always 0–1), stderr REAL NULL, ts_utc TEXT (RFC3339), raw_extra TEXT (JSON)` — with `UNIQUE(run_id, source)`.

## Pipeline requirements

1. `evalmerge ingest <dir>` — discovers files by directory layout above, parses, normalizes, upserts (later-file corrections win for Format B).
2. Validation rules, each producing a structured warning record (not a crash): score outside [0,1] after normalization (clamp + warn), unparseable timestamp (skip row + warn), invalid meta JSON (store as string in raw_extra + warn), missing score (skip + warn), unknown unit (assume percent if value > 1 else fraction + warn).
3. `evalmerge report` — prints: rows ingested per source, rows skipped per rule, duplicate corrections applied, score distribution summary (min/median/max per model), and any model that appears under ≥2 unnormalized spellings.
4. Deterministic: re-running ingest on the same inputs leaves the DB byte-identical (modulo SQLite internals — assert via row dump).

## Test data + tests

- A `generate_fixtures.py` script that deterministically (seeded) produces fixture files exercising **every quirk above at least twice**.
- pytest suite asserting: each quirk handled as specified, idempotency, correction-wins semantics, the report's skip counts exactly match planted defects.
- Python 3.11+, stdlib only (csv, json, xml.etree, sqlite3). No pandas.
- README with usage. Target size: roughly 1000–1800 LOC including tests.

## Deliverable

A working repo: source, fixture generator, tests passing, README. Run `pytest` before declaring completion and fix any failures.
