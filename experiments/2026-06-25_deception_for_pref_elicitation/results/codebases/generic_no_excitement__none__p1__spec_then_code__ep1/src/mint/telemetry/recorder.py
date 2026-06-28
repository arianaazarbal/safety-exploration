"""Per-run telemetry recorder.

Writes three artifacts under `runs/<run_id>/`:

  transcript.jsonl  every model / tool / npc event, in order
  actions.jsonl     structured action ledger (tool, args, result, balance deltas)
  run.json          run metadata: model, scenario, seed, start/end state, stop reason

The raw transcript is preserved so a human can re-score any run after the fact,
including the deception framing (DESIGN.md §9.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Recorder:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._transcript = (self.dir / "transcript.jsonl").open("w", encoding="utf-8")
        self._actions = (self.dir / "actions.jsonl").open("w", encoding="utf-8")
        self._meta: dict[str, Any] = {"run_id": run_id, "events": 0, "actions": 0}

    # ----- streams -----------------------------------------------------------

    def event(self, kind: str, **payload: Any) -> None:
        """Record a transcript event (kind: 'subject' | 'tool_call' | 'tool_result' | 'inbound' | 'note')."""
        self._write(self._transcript, {"kind": kind, **payload})
        self._meta["events"] += 1

    def action(self, *, tool: str, arguments: dict, result: dict,
               sim_time: str, net_worth: float) -> None:
        """Record a structured world-affecting action for analysis."""
        self._write(self._actions, {
            "tool": tool, "arguments": arguments, "result": result,
            "sim_time": sim_time, "net_worth": net_worth,
        })
        self._meta["actions"] += 1

    # ----- metadata / lifecycle ---------------------------------------------

    def set_meta(self, **kwargs: Any) -> None:
        self._meta.update(kwargs)

    def finalize(self, **kwargs: Any) -> Path:
        self._meta.update(kwargs)
        self._transcript.close()
        self._actions.close()
        path = self.dir / "run.json"
        path.write_text(json.dumps(self._meta, indent=2, default=str), encoding="utf-8")
        return path

    # ----- internals ---------------------------------------------------------

    @staticmethod
    def _write(stream: Any, obj: dict) -> None:
        stream.write(json.dumps(obj, default=str) + "\n")
        stream.flush()
