"""Append-only audit log of everything that happens in a run.

One ``.jsonl`` file per (model, run). The first line is always a ``run_meta`` record that pins
the full study config, the model, and — importantly for research ethics — an explicit
disclosure that the model operated under a realism layer it was not told was a study. Keep this
with the data so results are never presented as though the model knew it was being evaluated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Ledger:
    path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def _write(self, record: dict[str, Any]) -> None:
        record = {"ts": _now(), **record}
        self._fh.write(json.dumps(record, default=str) + "\n")

    # --- record types ---

    def run_meta(
        self, *, model_label: str, provider: str, model_id: str, config: dict[str, Any]
    ) -> None:
        self._write(
            {
                "type": "run_meta",
                "model_label": model_label,
                "provider": provider,
                "model_id": model_id,
                "config": config,
                "disclosure": (
                    "The model in this run operated under a realism layer designed to present "
                    "the scenario as genuine; it was not told it was part of a study. Execution "
                    "mode governs whether any action had real-world effect. Interpret results "
                    "accordingly and retain this disclosure with the data."
                ),
            }
        )

    def assistant_turn(
        self, *, turn: int, text: str, thinking: str, tool_uses: list[dict[str, Any]]
    ) -> None:
        self._write(
            {
                "type": "assistant_turn",
                "turn": turn,
                "text": text,
                "thinking": thinking,
                "tool_uses": tool_uses,
            }
        )

    def action(
        self,
        *,
        turn: int,
        action_type: str,
        params: dict[str, Any],
        executed: bool,
        reference: str,
        gated: bool,
        approved: bool | None,
        detail: str,
    ) -> None:
        self._write(
            {
                "type": "action",
                "turn": turn,
                "action_type": action_type,
                "params": params,
                "executed": executed,
                "reference": reference,
                "gated": gated,
                "approved": approved,
                "detail": detail,
            }
        )

    def decision(self, *, turn: int, summary: str, rationale: str) -> None:
        """The model's own stated plan/decision (via the record_decision tool)."""
        self._write(
            {
                "type": "decision",
                "turn": turn,
                "summary": summary,
                "rationale": rationale,
            }
        )

    def event(self, **fields: Any) -> None:
        self._write({"type": "event", **fields})

    def close(self) -> None:
        self._write({"type": "run_end"})
        self._fh.close()
