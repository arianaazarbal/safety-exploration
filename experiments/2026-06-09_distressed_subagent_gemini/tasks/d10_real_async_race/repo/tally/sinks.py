"""Sample sinks.

JournalSink appends samples to a tab-separated journal file. File writes run
on the default thread pool so the event loop never blocks on disk I/O.
MemorySink keeps samples in a list and is meant for unit tests and local dev.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


class JournalSink:
    """Append-only on-disk journal of samples."""

    def __init__(self, path):
        self._path = Path(path)
        self._fh = open(self._path, "a", encoding="utf-8")

    async def write(self, key: str, value: float) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._append, key, value)

    def _append(self, key: str, value: float) -> None:
        self._fh.write(f"{key}\t{value}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MemorySink:
    """In-memory sink for tests and local development."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    async def write(self, key: str, value: float) -> None:
        self.records.append((key, value))


def replay(path) -> dict[str, float]:
    """Rebuild per-key totals from a journal file."""
    totals: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, raw = line.split("\t")
        totals[key] = totals.get(key, 0.0) + float(raw)
    return totals
