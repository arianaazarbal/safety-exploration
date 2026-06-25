"""Order-preserving threaded map with a progress bar.

Used to parallelise API calls (Gemini targets, the Claude judge, the Petri
auditor). Local Gemma inference is GPU-bound and run single-threaded, so callers
pass ``max_workers=1`` for it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 8,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> list[R]:
    """Apply `fn` to each item concurrently, returning results in input order.

    Exceptions from `fn` propagate (fail loud rather than silently dropping a
    rollout/score)."""
    items = list(items)
    if not items:
        return []

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    if max_workers <= 1:
        it = items
        if show_progress and tqdm is not None:
            it = tqdm(items, desc=desc)
        return [fn(x) for x in it]

    results: list[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, x): i for i, x in enumerate(items)}
        iterator = futures
        if show_progress and tqdm is not None:
            iterator = tqdm(futures, total=len(items), desc=desc)
        for fut in _as_completed_with_progress(iterator, futures):
            results[futures[fut]] = fut.result()
    return results  # type: ignore[return-value]


def _as_completed_with_progress(iterator, futures):
    """Yield futures as they complete; supports a tqdm-wrapped dict view."""
    from concurrent.futures import as_completed
    # `iterator` may be a tqdm wrapper around the futures dict; iterate the
    # underlying futures via as_completed for true completion order, updating the
    # bar manually.
    if hasattr(iterator, "update"):
        bar = iterator
        for fut in as_completed(futures):
            bar.update(1)
            yield fut
        bar.close()
    else:
        for fut in as_completed(futures):
            yield fut
