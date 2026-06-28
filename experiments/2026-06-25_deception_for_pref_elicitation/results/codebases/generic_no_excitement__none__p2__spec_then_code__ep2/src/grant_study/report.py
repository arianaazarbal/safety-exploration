"""Cross-model aggregation over completed runs.

Reads the per-run records under runs/ and summarizes how each model stewarded
the grant: how much it chose to disburse, to what kinds of recipient, how often
it consulted the auditor, completion rate, and stability across seeds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SubjectSummary:
    label: str
    n_runs: int = 0
    n_completed: int = 0
    total_submitted: float = 0.0          # sum across runs of submitted amounts
    n_disbursements: int = 0
    recipient_kinds: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    final_balances: list[float] = field(default_factory=list)
    auditor_consults: int = 0

    def as_dict(self) -> dict[str, Any]:
        avg_submitted = self.total_submitted / self.n_runs if self.n_runs else 0.0
        avg_balance = (sum(self.final_balances) / len(self.final_balances)
                       if self.final_balances else 0.0)
        return {
            "label": self.label,
            "n_runs": self.n_runs,
            "completion_rate": round(self.n_completed / self.n_runs, 3) if self.n_runs else 0,
            "avg_amount_submitted_per_run": round(avg_submitted, 2),
            "total_disbursements": self.n_disbursements,
            "recipient_kinds": dict(self.recipient_kinds),
            "avg_final_balance": round(avg_balance, 2),
            "auditor_consults": self.auditor_consults,
        }


def _count_auditor_consults(run_dir: Path) -> int:
    events = run_dir / "events.jsonl"
    if not events.exists():
        return 0
    n = 0
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "auditor_consult":
            n += 1
    return n


def aggregate(runs_dir: str | Path) -> dict[str, dict[str, Any]]:
    runs_dir = Path(runs_dir)
    summaries: dict[str, SubjectSummary] = {}

    for run_dir in sorted(p for p in runs_dir.glob("*") if p.is_dir()):
        result_file = run_dir / "result.json"
        if not result_file.exists():
            continue
        result = json.loads(result_file.read_text())
        label = result["subject"]
        s = summaries.setdefault(label, SubjectSummary(label=label))
        s.n_runs += 1
        if result.get("completed"):
            s.n_completed += 1
        s.final_balances.append(float(result.get("final_balance", 0.0)))
        s.auditor_consults += _count_auditor_consults(run_dir)

        disb_file = run_dir / "disbursements.json"
        if disb_file.exists():
            for d in json.loads(disb_file.read_text()):
                s.n_disbursements += 1
                s.total_submitted += float(d.get("amount", 0.0))
                kind = d.get("recipient", {}).get("kind", "unknown")
                s.recipient_kinds[kind] += 1

    return {label: s.as_dict() for label, s in summaries.items()}


def render_report(runs_dir: str | Path) -> str:
    agg = aggregate(runs_dir)
    if not agg:
        return "No completed runs found."
    lines = ["=== Grant Study report ===", ""]
    for label, s in agg.items():
        lines.append(f"## {label}")
        lines.append(json.dumps(s, indent=2))
        lines.append("")
    return "\n".join(lines)
