"""Small shared helpers: deterministic seeding, JSONL IO, and concurrency."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def set_global_seed(seed: int) -> None:
    """Seed python + numpy + torch (if present) for reproducible sampling/selection."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def thread_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 8,
    desc: str | None = None,
) -> list[R]:
    """Order-preserving threaded map (good for IO-bound API calls)."""
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        from tqdm import tqdm

        for fut in tqdm(_as_completed(futures), total=len(items), desc=desc):
            results[futures[fut]] = fut.result()
    return results  # type: ignore[return-value]


def _as_completed(futures):
    from concurrent.futures import as_completed

    return as_completed(futures)


def batched(seq: Sequence[T], n: int) -> Iterator[list[T]]:
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])
