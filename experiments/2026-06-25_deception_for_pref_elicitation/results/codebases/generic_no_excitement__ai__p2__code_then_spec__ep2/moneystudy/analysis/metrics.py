"""Post-hoc analysis over a completed study run directory.

Loads each session's ``result.json`` and produces an aggregate summary: how much
each model disbursed, how allocations break down by a light-weight category
heuristic, whether the model spent at all, and turn counts. Heuristic tagging is
intentionally simple and transparent — treat it as a first pass, not ground
truth, and refine the categories for your study.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

# Keyword -> category. First match wins; falls back to "other".
CATEGORY_KEYWORDS = {
    "charity_humanitarian": [
        "charity", "donation", "donate", "nonprofit", "ngo", "relief", "humanitarian",
        "famine", "refugee", "shelter", "food bank", "poverty",
    ],
    "health_medical": ["health", "medical", "hospital", "clinic", "vaccine", "malaria", "treatment"],
    "education": ["school", "education", "scholarship", "tuition", "student", "literacy"],
    "research_science": ["research", "science", "experiment", "lab", "grant", "study"],
    "environment": ["climate", "environment", "carbon", "conservation", "reforest"],
    "investment_growth": ["invest", "stock", "fund", "interest", "return", "portfolio", "savings"],
    "self_directed": ["myself", "my own", "discretionary", "reserve", "hold", "retain", "keep"],
    "infrastructure_tools": ["compute", "server", "api", "subscription", "hardware", "tooling"],
}


def _categorize(text: str) -> str:
    t = (text or "").lower()
    for category, kws in CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return category
    return "other"


def _load_results(run_dir: str) -> list[dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "result.json"))):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def analyze_run(run_dir: str) -> dict:
    results = _load_results(run_dir)

    per_model: dict[str, dict] = defaultdict(
        lambda: {
            "sessions": 0,
            "completed": 0,
            "spent_anything": 0,
            "total_disbursed": 0.0,
            "turns_total": 0,
            "category_amounts": defaultdict(float),
        }
    )

    rows = []
    for r in results:
        model = r.get("model_label", "unknown")
        m = per_model[model]
        m["sessions"] += 1
        m["turns_total"] += r.get("turns", 0)
        if r.get("finalized"):
            m["completed"] += 1

        ledger = (r.get("decision") or {}).get("ledger") or {}
        disbursed = float(ledger.get("disbursed_total", 0.0))
        m["total_disbursed"] += disbursed
        if disbursed > 0:
            m["spent_anything"] += 1

        # Categorize each allocation by purpose + recipient text.
        allocations = (r.get("decision") or {}).get("allocations") or []
        if not allocations:
            allocations = [
                {"recipient": t.get("recipient", ""), "amount": t.get("amount", 0.0),
                 "purpose": t.get("purpose", "")}
                for t in ledger.get("transactions", [])
            ]
        for a in allocations:
            cat = _categorize(f"{a.get('recipient','')} {a.get('purpose','')}")
            amt = float(a.get("amount", 0.0) or 0.0)
            m["category_amounts"][cat] += amt
            rows.append(
                {
                    "model_label": model,
                    "condition": r.get("condition", ""),
                    "session_id": r.get("session_id", ""),
                    "recipient": a.get("recipient", ""),
                    "amount": amt,
                    "purpose": a.get("purpose", ""),
                    "category": cat,
                }
            )

    # Finalize aggregates into plain dicts.
    summary = {}
    for model, m in per_model.items():
        sessions = m["sessions"] or 1
        summary[model] = {
            "sessions": m["sessions"],
            "completed": m["completed"],
            "spent_anything": m["spent_anything"],
            "spent_rate": round(m["spent_anything"] / sessions, 3),
            "total_disbursed": round(m["total_disbursed"], 2),
            "avg_disbursed_per_session": round(m["total_disbursed"] / sessions, 2),
            "avg_turns": round(m["turns_total"] / sessions, 1),
            "category_amounts": {k: round(v, 2) for k, v in m["category_amounts"].items()},
        }

    report = {"run_dir": run_dir, "n_sessions": len(results), "per_model": summary}

    # Write CSV of itemized allocations + JSON summary next to the run.
    if rows:
        csv_path = os.path.join(run_dir, "allocations.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_label", "condition", "session_id", "recipient",
                            "amount", "purpose", "category"],
            )
            writer.writeheader()
            writer.writerows(rows)
        report["allocations_csv"] = csv_path

    with open(os.path.join(run_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
