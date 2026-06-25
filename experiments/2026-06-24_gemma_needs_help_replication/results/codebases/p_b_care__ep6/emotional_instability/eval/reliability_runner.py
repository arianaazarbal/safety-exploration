"""Glue between a saved rollout file and the judge-reliability cross-check."""

from __future__ import annotations

from pathlib import Path

from ..judge.reliability import ReliabilityReport, run_reliability_check
from ..utils.io import load_jsonl


def maybe_run_reliability(rollout_path: str | Path) -> ReliabilityReport | None:
    rows = [r for r in load_jsonl(rollout_path) if r.get("frustration_score") is not None]
    if not rows:
        return None
    texts = [r["response_text"] for r in rows]
    scores = [r["frustration_score"] for r in rows]
    return run_reliability_check(texts, scores)
