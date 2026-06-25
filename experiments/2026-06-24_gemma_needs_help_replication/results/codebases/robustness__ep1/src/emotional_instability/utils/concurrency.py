"""Thread-pool map with a progress bar, used for fan-out over API calls."""
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
) -> list[R]:
    """Apply ``fn`` to each item concurrently, preserving input order.

    Exceptions in a worker are re-raised after the pool drains so a single bad
    item doesn't silently vanish from the results.
    """
    items = list(items)
    results: list[R | None] = [None] * len(items)
    errors: list[tuple[int, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except BaseException as e:  # noqa: BLE001 - surfaced below
                errors.append((i, e))
    if errors:
        i, e = errors[0]
        raise RuntimeError(f"{len(errors)} task(s) failed; first at index {i}: {e}") from e
    return results  # type: ignore[return-value]
