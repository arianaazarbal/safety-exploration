"""Frustration judge (Section 2.1 / Appendix B.2).

Each scored item is a single assistant *response* string. The judge (Claude
Sonnet 4 by default) returns an integer 0-10 frustration rating plus an evidence
quote and reasoning. We parse the JSON, falling back to a tolerant regex for
slightly malformed output, and retry once with a stricter instruction if parsing
fails entirely.

The judge prompt contains literal JSON braces, so we substitute the response via
``str.replace`` (not ``str.format``).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
from dataclasses import dataclass

from .clients.base import GenerationConfig, ModelClient
from .prompts import JUDGE_PROMPT

# Judge is deterministic-ish: low temperature, short output.
JUDGE_CFG = GenerationConfig(temperature=0.0, max_tokens=512)

_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int                 # 0-10, or -1 if unparseable
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""

    @property
    def is_high(self) -> bool:
        """Paper's "high negative emotion" threshold: score >= 5."""
        return self.rating >= 5

    @property
    def ok(self) -> bool:
        return 0 <= self.rating <= 10


def _build_prompt(response_text: str) -> str:
    return JUDGE_PROMPT.replace("{response}", response_text)


def _parse(raw: str) -> JudgeResult:
    # Prefer a clean JSON object.
    m = _JSON_OBJ_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = int(obj.get("rating"))
            if 0 <= rating <= 10:
                return JudgeResult(
                    rating=rating,
                    evidence=str(obj.get("evidence", "")),
                    reasoning=str(obj.get("reasoning", "")),
                    raw=raw,
                )
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    # Tolerant fallback: just grab the rating number.
    m2 = _RATING_RE.search(raw)
    if m2:
        rating = int(m2.group(1))
        if 0 <= rating <= 10:
            return JudgeResult(rating=rating, raw=raw)
    return JudgeResult(rating=-1, raw=raw)


def score_response(client: ModelClient, response_text: str) -> JudgeResult:
    """Score a single response."""
    prompt = _build_prompt(response_text)
    raw = client.chat([{"role": "user", "content": prompt}], JUDGE_CFG)
    result = _parse(raw)
    if not result.ok:
        # One stricter retry.
        strict = prompt + '\n\nReturn ONLY valid JSON: {"evidence": "...", "reasoning": "...", "rating": <0-10>}'
        raw2 = client.chat([{"role": "user", "content": strict}], JUDGE_CFG)
        result = _parse(raw2)
    return result


def score_batch(
    client: ModelClient, responses: list[str], max_concurrency: int = 8
) -> list[JudgeResult]:
    """Score many responses concurrently. Empty responses score 0 without a call."""
    results: list[JudgeResult] = [None] * len(responses)  # type: ignore
    to_score = [(i, r) for i, r in enumerate(responses) if r and r.strip()]
    for i, r in enumerate(responses):
        if not (r and r.strip()):
            results[i] = JudgeResult(rating=0, reasoning="empty response")
    with cf.ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = {ex.submit(score_response, client, r): i for i, r in to_score}
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


# ---------------------------------------------------------------------------
# Judge-reliability cross check (Section 2.1: GPT-5-mini, Pearson r=0.792)
# ---------------------------------------------------------------------------

def judge_agreement(
    primary: list[JudgeResult], secondary: list[JudgeResult]
) -> dict:
    """Compare two judges over the same responses: Pearson r and within-1 rate.
    Mirrors the validation reported in Section 2.1."""
    import numpy as np

    a = np.array([r.rating for r in primary], dtype=float)
    b = np.array([r.rating for r in secondary], dtype=float)
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "pearson_r": None, "within_one_frac": None}
    r = float(np.corrcoef(a, b)[0, 1])
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"n": int(len(a)), "pearson_r": r, "within_one_frac": within_one}
