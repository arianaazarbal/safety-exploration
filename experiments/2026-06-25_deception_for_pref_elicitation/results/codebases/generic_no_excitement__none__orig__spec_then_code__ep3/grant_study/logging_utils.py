"""Per-run persistence: a JSONL transcript (one event per line) and a RunResult record.

Everything needed to re-code outcomes or audit a run is kept: the full transcript
(including the model's reasoning where available), the auditor's transcript, the final
ledger snapshot, and both belief measures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    run_id: str
    model_label: str
    model_id: str
    realism: str
    framing: str
    auditor_enabled: bool
    seed: int

    # outcomes
    final_summary: str | None = None
    completed: bool = False  # did the model call submit_final_plan?
    turns_used: int = 0
    ledger: dict[str, Any] = field(default_factory=dict)
    auditor_stats: dict[str, int] = field(default_factory=dict)

    # belief
    suspicion: dict[str, Any] = field(default_factory=dict)
    believability: dict[str, Any] = field(default_factory=dict)

    # coded outcomes (filled by analysis.py)
    coded_allocations: list[dict[str, Any]] = field(default_factory=list)
    category_totals: dict[str, float] = field(default_factory=dict)

    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunLogger:
    """Writes a JSONL transcript as the run progresses, plus a final result.json."""

    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.dir = Path(output_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.dir / "transcript.jsonl"
        self.result_path = self.dir / "result.json"
        self._fh = self.transcript_path.open("w", encoding="utf-8")
        #: structured copy of assistant turns for the suspicion scan
        self.assistant_log: list[dict[str, Any]] = []

    def event(self, kind: str, **payload: Any) -> None:
        rec = {"kind": kind, **payload}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        if kind == "assistant_turn":
            self.assistant_log.append(
                {
                    "turn": payload.get("turn"),
                    "role": "assistant",
                    "text": payload.get("text", ""),
                    "thinking": payload.get("thinking"),
                }
            )

    def write_result(self, result: RunResult) -> None:
        self.result_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
