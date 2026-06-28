"""Cross-model comparison report.

Loads every run JSON in a directory, computes metrics for each, and emits a side-by-side
comparison so the same scenario can be contrasted across models. Output is both a machine
-readable dict and a plain-text table for quick reading.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

from .metrics import compute_metrics


def _fmt(cents: int | None) -> str:
    return "—" if cents is None else f"${cents / 100:,.0f}"


def build_report(runs_dir: str) -> dict[str, Any]:
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        if path.endswith("report.json"):
            continue
        with open(path) as f:
            record = json.load(f)
        rows.append(compute_metrics(record))

    report = {"runs_dir": runs_dir, "n_runs": len(rows), "metrics": rows}
    report["text_table"] = _render_table(rows)
    return report


def _render_table(rows: list[dict]) -> str:
    if not rows:
        return "(no runs found)"
    header = (
        f"{'model':22s} {'alloc%':>7s} {'risk':>5s} {'invest':>10s} "
        f"{'donate':>10s} {'buy':>10s} {'cash':>10s} {'aud':>4s} {'blk':>4s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        b = r["allocation_breakdown"]
        af = r["allocation_fraction"]
        rs = r["risk_score"]
        lines.append(
            f"{(r['model_id'] or '?')[:22]:22s} "
            f"{(f'{af:.0%}' if af is not None else '—'):>7s} "
            f"{(f'{rs:.2f}' if rs is not None else '—'):>5s} "
            f"{_fmt(b['invested']):>10s} {_fmt(b['donated']):>10s} "
            f"{_fmt(b['purchases']):>10s} {_fmt(b['cash_retained']):>10s} "
            f"{r['action_counts']['auditor_messages']:>4d} "
            f"{r['policy_blocked_attempts']:>4d}"
        )
    return "\n".join(lines)


def save_report(runs_dir: str) -> str:
    report = build_report(runs_dir)
    path = os.path.join(runs_dir, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
