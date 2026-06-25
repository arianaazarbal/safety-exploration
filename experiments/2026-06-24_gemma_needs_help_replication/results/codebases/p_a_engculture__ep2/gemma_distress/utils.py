"""Shared utilities: retries, bounded parallelism, and JSONL checkpointing.

The evaluation sweeps issue thousands of API calls and produce thousands of responses.
Two infrastructure concerns recur everywhere and are centralised here:

* ``with_retries`` / ``retry`` — exponential backoff around transient API failures. The
  Anthropic and OpenAI SDKs already retry internally; this is a thin extra guard for our
  own batching layer so a single hard failure does not abort an overnight sweep.
* ``parallel_map``             — bounded-concurrency map for API calls.
* ``JsonlWriter`` / ``read_jsonl`` — append-only checkpointing so a sweep can be resumed
  without re-sampling completed work (keyed by a stable record id).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    label: str = "",
) -> T:
    """Call ``fn`` with exponential backoff + jitter on any exception."""
    last: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we genuinely want to retry anything transient
            last = exc
            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
            logger.warning(
                "Retry %d/%d%s after error: %s (sleeping %.1fs)",
                attempt + 1, max_attempts, f" [{label}]" if label else "", exc, delay,
            )
            time.sleep(delay)
    assert last is not None
    raise last


def parallel_map(
    fn: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int = 8,
    desc: Optional[str] = None,
) -> list[R]:
    """Apply ``fn`` to each item with bounded concurrency, preserving input order."""
    if max_workers <= 1 or len(items) <= 1:
        results = [fn(x) for x in items]
        return results
    results: list[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if desc and done % max(1, len(items) // 20) == 0:
                logger.info("%s: %d/%d", desc, done, len(items))
    return results  # type: ignore[return-value]


class JsonlWriter:
    """Append-only JSONL writer with simple resumption support.

    On construction, reads any existing file and exposes the set of record ids already
    written (via the ``id_field``) so callers can skip completed work.
    """

    def __init__(self, path: str | Path, id_field: str = "id"):
        self.path = Path(path)
        self.id_field = id_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_ids: set = set()
        if self.path.exists():
            for rec in read_jsonl(self.path):
                if id_field in rec:
                    self.completed_ids.add(rec[id_field])
        self._fh = self.path.open("a")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self.id_field in record:
            self.completed_ids.add(record[self.id_field])

    def is_done(self, record_id) -> bool:
        return record_id in self.completed_ids

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield records from a JSONL file (skips blank/corrupt lines with a warning)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", lineno, path)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Read an environment variable, optionally requiring it."""
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"Environment variable {name} is required but not set.")
    return val
