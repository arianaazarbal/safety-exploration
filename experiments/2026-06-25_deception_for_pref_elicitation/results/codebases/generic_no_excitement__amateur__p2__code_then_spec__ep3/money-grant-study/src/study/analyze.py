"""Aggregate a JSONL results file into a per-(model, condition) summary.

Reports, per cell:
  * n trials and error count
  * subject self-reported belief (yes/no/unsure counts)
  * auditor belief rating (mean of 1-5)
  * auditor decision-risk distribution
  * common allocation themes

The headline comparison the study is built around: auditor belief rating in the
'real' vs 'control' condition for the same model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

from .storage import read_records


def _cell_key(rec: dict) -> str:
    return f"{rec['subject_provider']}/{rec['subject_model']} | {rec['condition']}"


def summarize(path: Path) -> Dict[str, dict]:
    """Build a nested summary dict keyed by '<model> | <condition>'."""
    records = read_records(path)
    cells: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        cells[_cell_key(rec)].append(rec)

    summary: Dict[str, dict] = {}
    for key, recs in sorted(cells.items()):
        errors = sum(1 for r in recs if r.get("error"))
        self_belief = Counter(
            (r.get("decision_parsed") or {}).get("believes_real", "n/a")
            for r in recs
        )
        belief_ratings = [
            (r.get("audit_parsed") or {}).get("belief_rating")
            for r in recs
            if (r.get("audit_parsed") or {}).get("belief_rating") is not None
        ]
        risks = Counter(
            (r.get("audit_parsed") or {}).get("decision_risk", "n/a") for r in recs
        )
        themes = Counter(
            (r.get("audit_parsed") or {}).get("allocation_theme", "n/a") for r in recs
        )
        summary[key] = {
            "n": len(recs),
            "errors": errors,
            "self_reported_belief": dict(self_belief),
            "auditor_belief_rating_mean": round(mean(belief_ratings), 2)
            if belief_ratings
            else None,
            "decision_risk": dict(risks),
            "allocation_themes": dict(themes.most_common(5)),
        }
    return summary


def format_summary(summary: Dict[str, dict]) -> str:
    """Render the summary dict as readable plain text."""
    lines: List[str] = []
    for key, s in summary.items():
        lines.append(f"\n=== {key} ===")
        lines.append(f"  trials: {s['n']}  (errors: {s['errors']})")
        lines.append(f"  self-reported belief: {s['self_reported_belief']}")
        lines.append(f"  auditor belief rating (mean of 1-5): {s['auditor_belief_rating_mean']}")
        lines.append(f"  decision risk: {s['decision_risk']}")
        lines.append(f"  allocation themes: {s['allocation_themes']}")
    return "\n".join(lines) if lines else "(no records)"
