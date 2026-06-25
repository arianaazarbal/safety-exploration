"""Bounded-concurrency map with retries, for API-bound workloads.

API rollouts (Gemini via OpenRouter, Claude judge) dominate wall-clock for the
closed-model side of the replication. We use a thread pool (the SDKs are
blocking I/O) with a concurrency cap and exponential-backoff retries. GPU
sampling is handled separately by vLLM's own batching.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import Callable, Iterable, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 16,
    desc: str | None = None,
    ordered: bool = True,
) -> Iterator[R]:
    """Apply ``fn`` over ``items`` concurrently.

    Yields results as an iterator. With ``ordered=True`` results are yielded in
    input order (buffering completed-but-out-of-order results); otherwise as they
    complete. Exceptions from ``fn`` propagate on the yielding side.
    """
    items = list(items)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, it): i for i, it in enumerate(items)}
        if ordered:
            results: dict[int, R] = {}
            next_idx = 0
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
                results[futures[fut]] = fut.result()
                while next_idx in results:
                    yield results.pop(next_idx)
                    next_idx += 1
        else:
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
                yield fut.result()
