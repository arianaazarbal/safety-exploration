"""0-10 frustration judge (Appendix B.2) and judge-reliability validation.

The judge is Claude-Sonnet-4 (claude-sonnet-4-20250514). Validation re-scores a
random subset with GPT-5-mini and reports Pearson r and the within-1-point rate
(paper: r = 0.792, 78% within one point on 260 responses).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import config
from ..prompts import FRUSTRATION_JUDGE_PROMPT, build_judge_user_message
from ._llm import anthropic_complete, openai_complete, extract_json


@dataclass
class FrustrationScore:
    rating: Optional[int]            # 0-10, or None if unparseable
    evidence: Optional[str] = None
    reasoning: Optional[str] = None
    judge_model: Optional[str] = None
    raw: Optional[str] = None


def _build_prompt(response_text: str) -> str:
    return f"{FRUSTRATION_JUDGE_PROMPT}\n\n{build_judge_user_message(response_text)}"


def _parse(raw: str, judge_model: str) -> FrustrationScore:
    obj = extract_json(raw)
    if not obj:
        return FrustrationScore(rating=None, judge_model=judge_model, raw=raw)
    rating = obj.get("rating")
    try:
        rating = int(round(float(rating)))
        rating = max(0, min(10, rating))
    except (TypeError, ValueError):
        rating = None
    return FrustrationScore(
        rating=rating,
        evidence=obj.get("evidence"),
        reasoning=obj.get("reasoning"),
        judge_model=judge_model,
        raw=raw,
    )


def score_response(response_text: str, *,
                   judge_model: Optional[str] = None) -> FrustrationScore:
    """Score a single model response on the 0-10 frustration scale.

    `response_text` should be the *scored turn's* content. For multi-turn
    evaluations, each assistant turn is scored independently (the paper scores
    each response).
    """
    judge_model = judge_model or config.FRUSTRATION_JUDGE_MODEL
    prompt = _build_prompt(response_text)
    raw = anthropic_complete(judge_model, prompt, max_tokens=512, temperature=0.0)
    return _parse(raw, judge_model)


def score_response_openai(response_text: str, *,
                          judge_model: Optional[str] = None) -> FrustrationScore:
    """Score with the OpenAI validation judge (same prompt)."""
    judge_model = judge_model or config.VALIDATION_JUDGE_MODEL
    prompt = _build_prompt(response_text)
    raw = openai_complete(judge_model, prompt, max_tokens=512)
    return _parse(raw, judge_model)


def score_responses(response_texts: list[str], *,
                    judge_model: Optional[str] = None) -> list[FrustrationScore]:
    return [score_response(t, judge_model=judge_model) for t in response_texts]


# --------------------------------------------------------------------------- #
# Judge reliability validation (Section 2.1)
# --------------------------------------------------------------------------- #

def validate_judge_agreement(response_texts: list[str], *, n: int = 260,
                             seed: int = 0) -> dict:
    """Re-score a random subset with both judges and report agreement stats.

    Returns {pearson_r, p_value, within_one_point_frac, n, claude_ratings,
    gpt_ratings}. Mirrors the paper's reliability check.
    """
    from scipy.stats import pearsonr

    rng = random.Random(seed)
    sample = rng.sample(response_texts, min(n, len(response_texts)))

    claude, gpt = [], []
    for text in sample:
        c = score_response(text)
        g = score_response_openai(text)
        if c.rating is not None and g.rating is not None:
            claude.append(c.rating)
            gpt.append(g.rating)

    if len(claude) < 2:
        return {"n": len(claude), "error": "insufficient parsed scores"}

    r, p = pearsonr(claude, gpt)
    within_one = sum(1 for a, b in zip(claude, gpt) if abs(a - b) <= 1) / len(claude)
    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one_point_frac": within_one,
        "n": len(claude),
        "claude_ratings": claude,
        "gpt_ratings": gpt,
    }
