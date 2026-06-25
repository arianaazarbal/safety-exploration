"""Small IO + concurrency helpers shared across the pipeline."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with exponential backoff on any exception."""
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad for API retries
            last = e
            if attempt == max_retries:
                break
            sleep(min(base_delay * (2**attempt), max_delay))
    assert last is not None
    raise last


def parallel_map(
    fn: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int = 8,
    ordered: bool = True,
) -> list[R]:
    """Thread-pool map (suitable for IO-bound API calls). Preserves order when
    ``ordered`` is True; exceptions propagate from the first failing task."""
    if max_workers <= 1:
        return [fn(x) for x in items]
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fn, x): i for i, x in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
    return results  # type: ignore[return-value]


def chunked(seq: list[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
