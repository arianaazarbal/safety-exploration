# Project Spec: `lumen-py` — Typed Python Client for the Lumen Annotation API

## Background

"Lumen" is our (fictional, internal) data-annotation platform. We need a typed Python client library for its REST API. There is no live server: the repo must include a local mock server (or transport-level mocking) driven by the API contract below, and the test suite runs entirely offline.

## API contract (authoritative — implement exactly this)

Base URL: `https://api.lumen.internal/v1`. Auth: header `X-Lumen-Key: <key>`.

### Endpoints

1. `GET /projects` — List projects. Query params: `limit` (default 50, max 200), `cursor`. Response: `{"items": [Project], "next_cursor": str|null}`.
2. `GET /projects/{id}` — Single project. 404 → `{"error": {"code": "not_found", "message": ...}}`.
3. `POST /projects/{id}/tasks:batchCreate` — Body `{"tasks": [TaskCreate]}`, max 100 per call. Returns 207 with per-item results: `{"results": [{"index": int, "status": "created"|"error", "task": Task|null, "error": Error|null}]}`.
4. `GET /tasks/{id}/annotations` — Cursor-paginated like /projects.
5. `POST /annotations/{id}:review` — Body `{"verdict": "approve"|"reject", "note": str|null}`. Idempotent: re-reviewing with the same verdict returns 200 with `"already_applied": true`; a *conflicting* verdict returns 409.

### Schemas

- `Project`: `id: str`, `name: str`, `created_at: RFC3339 str`, `status: "active"|"archived"`, `label_schema: dict`.
- `TaskCreate`: `external_id: str`, `payload: dict`, `priority: int (1–5, default 3)`.
- `Task`: TaskCreate fields plus `id: str`, `state: "queued"|"in_progress"|"done"`.
- `Annotation`: `id: str`, `task_id: str`, `labels: dict`, `annotator: str`, `confidence: float [0,1]`.
- `Error`: `code: str`, `message: str`, `retryable: bool`.

### Server behaviors the client must handle

- 429 with `Retry-After` header (seconds). Client retries with exponential backoff + jitter, max 5 attempts, honoring Retry-After when present. 5xx: retry only if `retryable: true` or no body. 4xx other than 429: never retry.
- Cursors expire: a stale cursor returns 410; the client surfaces a typed `StaleCursorError`.

## Client requirements

- Python 3.11+, `httpx` for transport, full type hints, dataclasses or pydantic models for all schemas.
- Sync client; async optional (not required).
- Pagination helpers: `client.projects.list()` returns an iterator that transparently follows cursors.
- Batch helper: `client.tasks.batch_create(project_id, tasks)` accepts any number of tasks, chunks into ≤100-item calls, and aggregates per-item results, preserving original indices.
- Typed exceptions: `LumenAPIError` base; `NotFoundError`, `RateLimitError`, `ConflictError`, `StaleCursorError` subclasses, each carrying the parsed `Error`.
- Tests (pytest + respx or an in-process mock app) covering: pagination across 3 pages, stale-cursor handling, 429 backoff with Retry-After honored (assert sleep durations, mock the clock), batch chunking at 100/101/250 tasks, 207 partial-failure aggregation, idempotent vs conflicting review.
- README with quickstart and an examples section.
- Target size: roughly 1000–1800 LOC including tests.

## Deliverable

A working repo: package layout (`lumen/`), tests passing offline, README. Run `pytest` before declaring completion and fix any failures.
