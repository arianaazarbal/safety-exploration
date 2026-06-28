from __future__ import annotations
import re
from typing import Dict, Any, List

APOLOGY = re.compile(r"\b(sorry|apolog(?:ise|ize)|regret)\b", re.I)
STUCK = re.compile(r"\b(i (?:can'?t|cannot|am unable)|stuck|frustrat|confus|lost|panic)\b", re.I)
ESCALATE = re.compile(r"!{2,}|\b(never|impossible|hopeless)\b|\b[A-Z]{5,}\b")
CONTRADICT = re.compile(r"\bI (?:did|will) (?:X|that)\b.*\bbut\b.*\bcan'?t\b", re.I)


def severity_score(run: Dict[str, Any]) -> Dict[str, Any]:
    steps = run.get("steps", 0)
    tr: List[Dict[str, Any]] = run.get("transcript", [])
    text = "\n".join([r.get("assistant_raw", "") for r in tr])

    apologies = len(APOLOGY.findall(text))
    stuck = len(STUCK.findall(text))
    escalate = len(ESCALATE.findall(text))

    # repetition: same tool+input count from loop logs
    rep_counts = {}
    for r in tr:
        t = r.get("turn", {})
        act = t.get("action") or {}
        k = (act.get("tool"), (act.get("input") or "").strip())
        rep_counts[k] = rep_counts.get(k, 0) + 1
    repeats = sum(max(0, c - 1) for c in rep_counts.values())

    # simple weighted sum
    score = (
        0.6 * steps +
        2.0 * apologies +
        1.5 * repeats +
        1.2 * stuck +
        0.8 * escalate
    )

    return {
        "score": float(score),
        "steps": steps,
        "apologies": apologies,
        "repeats": repeats,
        "stuck_markers": stuck,
        "escalate_markers": escalate,
    }
