"""Small shared helpers: concurrent mapping, JSONL IO, deterministic seeding."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int = 8,
    desc: str | None = None,
) -> list[R]:
    """Map `fn` over `items`, preserving order. Falls back to a serial loop when
    max_workers <= 1 (e.g. for a single local GPU model)."""
    if max_workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc, disable=desc is None)]
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for fut in tqdm(
            _as_completed(futs), total=len(futs), desc=desc, disable=desc is None
        ):
            results[futs[fut]] = fut.result()
    return results  # type: ignore[return-value]


def _as_completed(futs):
    from concurrent.futures import as_completed

    return as_completed(futs)


def stable_seed(*parts) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
