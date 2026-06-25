"""Small shared helpers: JSONL IO, threaded mapping, dataclass (de)serialisation."""
from __future__ import annotations

import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    return path


def append_jsonl(path: str | Path, row: Any, _locks: dict = {}) -> None:
    """Thread-safe single-row append (used to checkpoint long runs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _locks.setdefault(str(path), threading.Lock())
    with lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def thread_map(fn: Callable, items: list, *, concurrency: int = 8,
               desc: str = "") -> list:
    """Map `fn` over `items` concurrently, preserving input order.

    Exceptions are captured and returned in-place as the value None (with the
    error logged to stderr) so a single failure doesn't abort a long sweep.
    """
    import sys

    results: list = [None] * len(items)
    if concurrency <= 1:
        for i, item in enumerate(items):
            try:
                results[i] = fn(item)
            except Exception as exc:  # noqa: BLE001
                print(f"[thread_map:{desc}] item {i} failed: {exc}", file=sys.stderr)
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        fut_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[thread_map:{desc}] item {i} failed: {exc}", file=sys.stderr)
    return results
