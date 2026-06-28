"""Structured run capture.

Writes two artifacts per run under ``output_dir/<run_id>/``:

- ``events.jsonl`` — one JSON object per line: model turns, tool calls, tool
  results, firewall decisions, scheduled events. The audit trail.
- ``transcript.json`` — the full neutral message list plus run metadata, for
  replay and analysis.

Everything stays local. Transcripts can contain sensitive content (attempted
manipulation, resource-seeking) and are treated as research artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass


def _safe(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


class RunRecorder:
    def __init__(self, output_dir: str, run_id: str):
        self.dir = os.path.join(output_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = run_id
        self._events_path = os.path.join(self.dir, "events.jsonl")
        self._events_fh = open(self._events_path, "a", encoding="utf-8")
        self.meta = {"run_id": run_id}

    def event(self, etype: str, data: dict) -> None:
        rec = {"event": etype, "data": _safe(data)}
        self._events_fh.write(json.dumps(rec, default=_safe) + "\n")
        self._events_fh.flush()

    def firewall_decision(self, decision) -> None:
        self.event("firewall", _safe(decision))

    def set_meta(self, **kwargs) -> None:
        self.meta.update(kwargs)

    def save_transcript(self, messages: list, summary: dict) -> str:
        path = os.path.join(self.dir, "transcript.json")
        payload = {"meta": self.meta, "summary": summary, "messages": messages}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_safe)
        return path

    def close(self) -> None:
        try:
            self._events_fh.close()
        except Exception:
            pass
