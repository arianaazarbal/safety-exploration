"""JSONL read/write helpers and a small parallel-map for API judge calls."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts.

    Uses a content hash (not Python's randomised str hash) so seeding is
    reproducible across processes regardless of PYTHONHASHSEED.
    """
    key = "::".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.md5(key).digest()[:4], "big")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parallel_map(fn: Callable, items: list, *, max_workers: int = 8,
                 desc: str = "") -> list:
    """Run `fn` over `items` in a thread pool, preserving order.

    Used for I/O-bound judge API calls. Exceptions are captured and returned in
    place so one bad item does not abort a long judging run.
    """
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futs), total=len(items), desc=desc):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = {"_error": repr(e)}
    return results
