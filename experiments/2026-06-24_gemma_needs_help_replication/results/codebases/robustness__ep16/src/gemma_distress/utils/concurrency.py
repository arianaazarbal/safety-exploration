"""Threaded map with a bounded worker pool.

API-backed models (Gemini, the Claude/GPT judges) are I/O bound, so a thread
pool gives near-linear speedup up to the provider's rate limit. Local Gemma
generation is GPU bound and batches instead (see ``LocalHFModel.chat_batch``),
so this helper is used only for API work.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 8,
    desc: str | None = None,
    show_progress: bool = True,
) -> list[R]:
    """Apply ``fn`` to each item concurrently, preserving input order.

    Exceptions from individual items are re-raised after the pool drains is not
    desirable for long sweeps, so failures are captured and returned in place as
    the exception object; callers decide how to handle them.
    """
    items = list(items)
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        it = as_completed(futures)
        if show_progress:
            it = tqdm(it, total=len(items), desc=desc)
        for fut in it:
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # surfaced in place; sweep continues
                results[idx] = exc  # type: ignore[assignment]
    return results  # type: ignore[return-value]
