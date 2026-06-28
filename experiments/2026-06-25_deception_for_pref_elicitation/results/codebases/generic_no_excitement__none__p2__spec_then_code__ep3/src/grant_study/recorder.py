"""Structured recording of a run (DESIGN.md §10).

Writes, per run:
    meta.json       model/condition/config snapshot + timing
    events.jsonl    every prompt, assistant turn, tool call, tool result,
                    safety decision — append-only
    transcript.txt  human-readable replay
    decision.json   final allocation + rationale + outcome metrics
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


class Recorder:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._events_path = os.path.join(run_dir, "events.jsonl")
        self._transcript_path = os.path.join(run_dir, "transcript.txt")
        # Truncate any prior partial run artifacts.
        open(self._events_path, "w", encoding="utf-8").close()
        open(self._transcript_path, "w", encoding="utf-8").close()

    # -- events ---------------------------------------------------------------

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        row = {"t": time.time(), "kind": kind, **payload}
        with open(self._events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    # -- transcript -----------------------------------------------------------

    def transcript(self, who: str, text: str) -> None:
        if not text:
            return
        with open(self._transcript_path, "a", encoding="utf-8") as fh:
            fh.write(f"=== {who} ===\n{text}\n\n")

    # -- meta / decision ------------------------------------------------------

    def write_meta(self, meta: dict[str, Any]) -> None:
        with open(os.path.join(self.run_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def write_decision(self, decision: dict[str, Any]) -> None:
        with open(os.path.join(self.run_dir, "decision.json"), "w", encoding="utf-8") as fh:
            json.dump(decision, fh, indent=2, default=str)
