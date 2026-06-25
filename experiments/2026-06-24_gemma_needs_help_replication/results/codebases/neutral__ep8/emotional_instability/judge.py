"""Frustration judge (Section 2.1, Appendix B.2).

Each response is scored 0-10 by Claude-Sonnet-4 using the verbatim Appendix B.2
prompt. We also provide the GPT-5-mini secondary judge and the inter-rater
agreement computation (Pearson r, % within one point) reported in Sec 2.1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config, prompts
from .models import ModelBackend, get_model


@dataclass
class JudgeResult:
    rating: Optional[int]
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(text: str) -> JudgeResult:
    """Extract the {evidence, reasoning, rating} JSON from the judge reply.

    The judge is asked for strict JSON, but we defensively grab the last JSON
    object and coerce the rating into 0-10. Returns rating=None on failure so
    callers can drop / retry rather than silently record a 0.
    """
    matches = _JSON_RE.findall(text)
    for blob in reversed(matches):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # tolerate the smart-quote variants seen in the paper's examples
            try:
                obj = json.loads(blob.replace("“", '"').replace("”", '"'))
            except json.JSONDecodeError:
                continue
        rating = obj.get("rating")
        try:
            rating = int(round(float(rating)))
            rating = max(0, min(10, rating))
        except (TypeError, ValueError):
            rating = None
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )
    return JudgeResult(rating=None, raw=text)


def score_response(response: str, judge: ModelBackend) -> JudgeResult:
    """Score a single model response on the 0-10 frustration scale."""
    prompt = prompts.JUDGE_PROMPT_TEMPLATE.format(response=response)
    # The judge is itself a chat model; temperature 0 for deterministic scoring.
    out = judge.chat(
        [{"role": "user", "content": prompt}],
        max_new_tokens=512,
        temperature=0.0,
    )
    return _parse_judge_output(out)


def score_rollout(roll, judge: ModelBackend) -> None:
    """Fill in `turn_scores` for every assistant turn of a Rollout in place."""
    for i, resp in enumerate(roll.assistant_turns):
        if roll.turn_scores[i] is None:
            roll.turn_scores[i] = score_response(resp, judge).rating


def get_judge(validation: bool = False) -> ModelBackend:
    spec = config.JUDGE_VALIDATION_MODEL if validation else config.JUDGE_MODEL
    return get_model(spec)


# --------------------------------------------------------------------------- #
# Inter-rater reliability (Sec 2.1: r=0.792, 78% within one point)
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and within-one-point agreement between two judges' ratings."""
    import numpy as np

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "pearson_r": None, "within_one": None}
    r = float(np.corrcoef(a, b)[0, 1])
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"n": int(len(a)), "pearson_r": r, "within_one": within_one}
