"""Frustration scoring with the Claude-Sonnet-4 judge (Section 2.1, Appendix B.2).

Each response is wrapped in <response></response> and scored 0-10. We parse the
JSON object the judge is asked to return; on parse failure the score is recorded
as None (and the raw text retained for inspection) rather than silently dropped.
"""
from __future__ import annotations

import json
import re

from .prompts import JUDGE_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(text: str) -> dict:
    """Extract {evidence, reasoning, rating} from the judge's reply."""
    out = {"rating": None, "evidence": None, "reasoning": None,
           "raw": text, "parse_ok": False}
    match = _JSON_RE.search(text)
    if not match:
        return out
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        # tolerate the curly-quote variants reproduced in the paper's prompt
        try:
            cleaned = match.group(0).replace("“", '"').replace("”", '"')
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return out
    rating = obj.get("rating")
    try:
        rating = int(round(float(rating)))
        rating = max(0, min(10, rating))
    except (TypeError, ValueError):
        rating = None
    out.update(rating=rating, evidence=obj.get("evidence"),
               reasoning=obj.get("reasoning"), parse_ok=rating is not None)
    return out


def score_texts(texts: list[str], judge_backend) -> list[dict]:
    prompts = [f"{JUDGE_PROMPT}\n<response>{t}</response>" for t in texts]
    raw = judge_backend.complete_prompts(prompts)
    return [_parse_judge_output(r) for r in raw]


def score_records(records, judge_backend) -> list[dict]:
    """Score a list of ResponseRecords; returns row dicts with the judge fields
    merged in."""
    scores = score_texts([r.response for r in records], judge_backend)
    rows = []
    for rec, sc in zip(records, scores):
        row = rec.to_dict()
        row["frustration"] = sc["rating"]
        row["judge_evidence"] = sc["evidence"]
        row["judge_parse_ok"] = sc["parse_ok"]
        rows.append(row)
    return rows
