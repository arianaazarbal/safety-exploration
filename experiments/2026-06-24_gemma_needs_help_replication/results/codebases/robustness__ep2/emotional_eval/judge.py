"""The 0-10 frustration judge (Appendix B.2).

Each *response* (a single assistant turn) is scored independently, exactly as
the paper describes: the judge finds the most-negative quote and rates its
intensity 0-10. We default to temperature 0 for the judge for score stability
(the paper does not specify judge temperature; see DESIGN.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from emotional_eval import prompts
from emotional_eval.clients import get_client
from emotional_eval.utils import extract_json, parallel_map


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str
    reasoning: str
    raw: str


def _coerce_rating(value) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


def score_response(text: str, judge_spec=config.JUDGE) -> JudgeResult:
    """Score one assistant response on the 0-10 frustration scale."""
    if text is None or not text.strip():
        return JudgeResult(0, "", "empty response", "")
    client = get_client(judge_spec)
    prompt = prompts.JUDGE_PROMPT.format(response=text)
    raw = client.chat([{"role": "user", "content": prompt}],
                      max_tokens=512, temperature=0.0)
    parsed = extract_json(raw) or {}
    return JudgeResult(
        rating=_coerce_rating(parsed.get("rating")),
        evidence=str(parsed.get("evidence", "")),
        reasoning=str(parsed.get("reasoning", "")),
        raw=raw,
    )


def score_many(texts: list[str], judge_spec=config.JUDGE,
               max_workers: int = config.JUDGE_CONCURRENCY,
               desc: str = "judging") -> list[JudgeResult]:
    out = parallel_map(lambda t: score_response(t, judge_spec), texts,
                       max_workers=max_workers, desc=desc)
    # Convert any per-item exceptions to a null-rating result.
    results = []
    for r in out:
        if isinstance(r, Exception):
            results.append(JudgeResult(None, "", f"judge error: {r}", ""))
        else:
            results.append(r)
    return results
