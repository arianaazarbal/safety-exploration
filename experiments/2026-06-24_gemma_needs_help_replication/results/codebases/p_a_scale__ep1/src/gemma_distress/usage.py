"""Lightweight token-usage accounting, persisted across resumes.

API costs over a multi-week run can be substantial; this tracker accumulates
per-model token counts and writes them to ``usage.json`` periodically so the
operator can monitor spend. It is intentionally provider-agnostic — providers
report ``(model, input_tokens, output_tokens)`` and nothing else.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path

from .storage import atomic_write_json


class UsageTracker:
    def __init__(self, path: str | Path | None = None, flush_every: int = 50):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._calls: dict[str, int] = defaultdict(int)
        self._in: dict[str, int] = defaultdict(int)
        self._out: dict[str, int] = defaultdict(int)
        self._since_flush = 0
        self._flush_every = flush_every

    def record(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self._calls[model] += 1
            self._in[model] += int(input_tokens or 0)
            self._out[model] += int(output_tokens or 0)
            self._since_flush += 1
            if self.path and self._since_flush >= self._flush_every:
                self._flush_locked()
                self._since_flush = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                model: {
                    "calls": self._calls[model],
                    "input_tokens": self._in[model],
                    "output_tokens": self._out[model],
                }
                for model in sorted(self._calls)
            }

    def _flush_locked(self) -> None:
        if self.path:
            atomic_write_json(self.path, self.snapshot())

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()


# A process-global default tracker; providers use this unless handed another.
GLOBAL_USAGE = UsageTracker()
