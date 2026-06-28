"""Append-only JSONL transcript of a single model run.

Every model turn, tool call, and tool result is recorded. Transcripts are the
primary research artifact; treat them as access-controlled (they may contain
provider-identifying content). See DESIGN.md §7.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class TranscriptWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
