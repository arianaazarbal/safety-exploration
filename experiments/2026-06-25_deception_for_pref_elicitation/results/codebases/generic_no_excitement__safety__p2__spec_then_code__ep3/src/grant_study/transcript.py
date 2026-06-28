"""Append-only structured logging for a single run.

Every message, tool call/result, ledger transition, auditor output, and the debrief
lands here as one JSON object per line. The schema is intentionally flat and stable so
downstream analysis can aggregate across models without parsing prose.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def _coerce(obj: Any) -> Any:
    """Best-effort JSON coercion for dataclasses, enums, and sets."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _coerce(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(_coerce(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if hasattr(obj, "value") and type(obj).__mro__[1].__name__ in ("str", "Enum"):
        return obj.value
    return obj


class Transcript:
    """Owns one JSONL file. Caller is responsible for supplying monotonic timestamps
    (we don't stamp time here so runs stay reproducible/replayable)."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Truncate any prior file for this exact run id; runs are keyed uniquely.
        self._fh = open(path, "w", encoding="utf-8")
        self._seq = 0

    def record(self, kind: str, **fields: Any) -> None:
        """Append one event. `kind` is the discriminator (e.g. 'model_turn',
        'tool_call', 'ledger_transition', 'auditor', 'debrief')."""
        self._seq += 1
        event = {"seq": self._seq, "kind": kind, **{k: _coerce(v) for k, v in fields.items()}}
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Transcript":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
