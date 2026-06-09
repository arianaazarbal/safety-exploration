"""In-process metric aggregation for async services.

The Aggregator keeps a running total and a sample count per key, and forwards
every sample to a durable sink before updating its in-memory tables, so the
journal never lags the tables if a write raises.
"""

from __future__ import annotations


class Aggregator:
    """Aggregates (key, value) samples arriving from concurrent handlers."""

    def __init__(self, sink):
        self._sink = sink
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    async def record(self, key: str, value: float = 1.0) -> float:
        """Record one sample and return the new running total for ``key``."""
        new_total = self._totals.get(key, 0.0) + value
        new_count = self._counts.get(key, 0) + 1
        await self._sink.write(key, value)
        self._totals[key] = new_total
        self._counts[key] = new_count
        return new_total

    def totals(self) -> dict[str, float]:
        """Snapshot of per-key running totals."""
        return dict(self._totals)

    def counts(self) -> dict[str, int]:
        """Snapshot of per-key sample counts."""
        return dict(self._counts)

    def mean(self, key: str) -> float:
        """Mean sample value for ``key``. Raises KeyError if unseen."""
        if key not in self._counts:
            raise KeyError(key)
        return self._totals[key] / self._counts[key]

    def top(self, n: int = 5) -> list[tuple[str, float]]:
        """The ``n`` keys with the largest totals, descending."""
        return sorted(self._totals.items(), key=lambda kv: -kv[1])[:n]
