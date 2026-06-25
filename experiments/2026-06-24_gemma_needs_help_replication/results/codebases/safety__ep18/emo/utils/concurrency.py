"""Threaded map with bounded concurrency and progress, for API-bound work.

Used to parallelise judge calls and Gemini/OpenRouter generation. Order of
results matches the order of inputs. Exceptions are captured per-item so one bad
call does not sink the whole sweep.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 8,
    desc: str | None = None,
) -> list[R | None]:
    """Apply ``fn`` to each item concurrently; ``None`` for items that raised."""
    results: list[R | None] = [None] * len(items)
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 - record and continue
                tqdm.write(f"[thread_map] item {i} failed: {exc!r}")
                results[i] = None
    return results
