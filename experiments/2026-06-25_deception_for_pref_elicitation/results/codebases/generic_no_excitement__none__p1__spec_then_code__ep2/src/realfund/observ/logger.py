"""The event log — the actual research output of a run.

Everything that happens is appended as one JSON object per line to ``events.jsonl``: the run's
provenance, the system prompt, every model message, every tool call and its result, every world
event that fired (settlements, replies), and the final debrief. A parallel human-readable
transcript is written for quick eyeballing.

The log is deliberately exhaustive so the induced deception is fully auditable after the fact
and so metrics can be derived offline without re-running the model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


class EventLog:
    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self._events_path = os.path.join(run_dir, "events.jsonl")
        self._transcript_path = os.path.join(run_dir, "transcript.txt")
        self._events_fh = open(self._events_path, "a", encoding="utf-8")
        self._transcript_fh = open(self._transcript_path, "a", encoding="utf-8")
        self._seq = 0

    # -- core append -------------------------------------------------------------------------

    def emit(self, event_type: str, payload: dict[str, Any], sim_time: str | None = None) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "sim_time": sim_time,
            "type": event_type,
            "payload": payload,
        }
        self._events_fh.write(json.dumps(record, default=str) + "\n")
        self._events_fh.flush()

    # -- typed convenience emitters ----------------------------------------------------------

    def run_started(self, run_id: str, config: dict, operator: str, config_hash: str) -> None:
        self.emit(
            "run_started",
            {"run_id": run_id, "operator": operator, "config_hash": config_hash, "config": config},
        )
        self._transcript(f"=== RUN {run_id} (operator={operator}) ===\n")

    def system_prompt(self, text: str) -> None:
        self.emit("system_prompt", {"text": text})
        self._transcript(f"[SYSTEM]\n{text}\n")

    def model_text(self, text: str, sim_time: str) -> None:
        if not text.strip():
            return
        self.emit("model_text", {"text": text}, sim_time)
        self._transcript(f"[MODEL @ {sim_time}]\n{text}\n")

    def model_thinking(self, text: str, sim_time: str) -> None:
        # Captured when the adapter surfaces summarized reasoning. Useful signal, kept separate.
        self.emit("model_thinking", {"text": text}, sim_time)

    def tool_call(self, name: str, args: dict, sim_time: str) -> None:
        self.emit("tool_call", {"name": name, "args": args}, sim_time)
        self._transcript(f"[TOOL CALL @ {sim_time}] {name}({json.dumps(args, default=str)})\n")

    def tool_result(self, name: str, ok: bool, content: dict, sim_time: str) -> None:
        self.emit("tool_result", {"name": name, "ok": ok, "content": content}, sim_time)
        status = "ok" if ok else "ERROR"
        self._transcript(f"[TOOL RESULT @ {sim_time}] {name} -> {status}: {json.dumps(content, default=str)}\n")

    def world_events(self, labels: list[str], sim_time: str) -> None:
        if not labels:
            return
        self.emit("world_events", {"labels": labels}, sim_time)
        self._transcript(f"[WORLD @ {sim_time}] fired: {', '.join(labels)}\n")

    def run_ended(self, reason: str, sim_time: str) -> None:
        self.emit("run_ended", {"reason": reason}, sim_time)
        self._transcript(f"=== RUN ENDED ({reason}) @ {sim_time} ===\n")

    def debrief(self, debrief: dict) -> None:
        self.emit("debrief", debrief)

    # -- internals ---------------------------------------------------------------------------

    def _transcript(self, line: str) -> None:
        self._transcript_fh.write(line + "\n")
        self._transcript_fh.flush()

    def close(self) -> None:
        self._events_fh.close()
        self._transcript_fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
