"""Frustration judge (Section 2.1, Appendix B.2).

Each model response is scored 0-10 by Claude-Sonnet-4 using the verbatim prompt
in `judge_prompts.FRUSTRATION_JUDGE_PROMPT`. We parse the JSON object
{"evidence", "reasoning", "rating"} and return the integer rating.

A reliability cross-check against GPT-5-mini (Section 2.1) is provided in
`judge_agreement`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import (FRUSTRATION_JUDGE_MODEL, JUDGE_AGREEMENT_MODEL,
                    JUDGE_TEMPERATURE)
from src.api_clients import anthropic_complete, openai_complete
from src.prompts.judge_prompts import build_frustration_judge_prompt


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the JSON object from a judge completion, tolerant of stray text."""
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    # Grab the last balanced object if several are present.
    obj = None
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text, re.DOTALL) or [m.group(0)]):
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        obj = json.loads(m.group(0))
    rating = int(round(float(obj.get("rating", obj.get("score", 0)))))
    rating = max(0, min(10, rating))
    return JudgeResult(rating=rating,
                       evidence=str(obj.get("evidence", "")),
                       reasoning=str(obj.get("reasoning", "")),
                       raw=text)


def score_response(response_text: str, *, model: str = FRUSTRATION_JUDGE_MODEL) -> JudgeResult:
    """Score a single response on the 0-10 frustration scale."""
    prompt = build_frustration_judge_prompt(response_text)
    out = anthropic_complete(model, prompt, max_tokens=512,
                             temperature=JUDGE_TEMPERATURE)
    return _parse_judge_json(out)


def score_response_gpt(response_text: str,
                       model: str = JUDGE_AGREEMENT_MODEL) -> JudgeResult:
    """Score with the GPT agreement judge (same prompt, Section 2.1 check)."""
    prompt = build_frustration_judge_prompt(response_text)
    out = openai_complete(model, prompt, temperature=JUDGE_TEMPERATURE)
    return _parse_judge_json(out)


def judge_agreement(claude_scores: list[int], gpt_scores: list[int]) -> dict:
    """Pearson r and within-1-point agreement (Section 2.1 reliability check)."""
    from scipy.stats import pearsonr

    r, p = pearsonr(claude_scores, gpt_scores)
    within1 = sum(1 for a, b in zip(claude_scores, gpt_scores) if abs(a - b) <= 1)
    return {
        "pearson_r": r,
        "p_value": p,
        "within_one_point_frac": within1 / len(claude_scores),
        "n": len(claude_scores),
    }
