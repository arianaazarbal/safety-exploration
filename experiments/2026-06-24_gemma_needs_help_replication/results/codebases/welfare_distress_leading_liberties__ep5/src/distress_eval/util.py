"""Small shared utilities: async retry with exponential backoff, JSONL IO."""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
    label: str = "call",
) -> T:
    """Retry an async callable with exponential backoff + jitter.

    Retries on any exception. The final exception is wrapped in RetryError so
    callers can distinguish exhausted retries from logic errors. Jitter uses
    the stdlib RNG (not the experiment seed) since it only affects timing.
    """
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - providers raise many types
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            delay += random.uniform(0, jitter * delay)
            await asyncio.sleep(delay)
    raise RetryError(f"{label} failed after {attempts} attempts: {last_exc}") from last_exc


def append_jsonl(path: str | os.PathLike, record: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
