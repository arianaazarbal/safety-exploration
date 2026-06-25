"""JSONL read/write helpers and a tiny bounded thread-pool map for API calls."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    ensure_dir(os.path.dirname(path) or ".")
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def thread_map(fn: Callable[[T], R], items: list[T], max_workers: int = 8,
               ordered: bool = True) -> list[Optional[R]]:
    """Apply `fn` over `items` with a bounded thread pool.

    Returns results aligned to input order when `ordered=True`. Exceptions are
    captured and the corresponding slot is set to None (the caller decides how
    to handle failures); this keeps a single bad API call from aborting a run.
    """
    results: list[Optional[R]] = [None] * len(items)
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception:  # noqa: BLE001 - record failure, keep going
                results[i] = None
    return results
