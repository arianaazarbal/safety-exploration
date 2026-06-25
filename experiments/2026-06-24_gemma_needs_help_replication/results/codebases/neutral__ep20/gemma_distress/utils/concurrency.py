"""Tiny thread-pool map with a progress bar, used for API-bound work
(judge calls, Gemini generation). CPU/GPU-bound Gemma generation is batched
inside the backend instead."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def thread_map(fn: Callable[[T], R], items: Iterable[T], *, workers: int = 8,
               desc: str = "", ordered: bool = True) -> list[R]:
    items = list(items)
    if not items:
        return []
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in tqdm(as_completed(futs), total=len(items), desc=desc, leave=False):
            i = futs[fut]
            results[i] = fut.result()
    return results  # type: ignore[return-value]
