"""The 0--10 frustration judge (Section 2.1 / Appendix B.2).

Each model response is scored independently by Claude Sonnet 4 using the verbatim
judge prompt. The judge returns JSON with the most-negative quote, its reasoning,
and an integer 0--10 rating. We also provide a GPT-5-mini cross-check judge and an
agreement computation to reproduce the inter-rater reliability check (Pearson
r = 0.792; 78% within one point).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import SamplingConfig
from .models import ChatMessage, ModelClient, build_client
from .prompts import judge_user_message

# Judges are run at temperature 0 for determinism (the paper does not specify a
# judge temperature; 0 is the standard choice for an LLM grader and is what we
# document in DESIGN.md).
JUDGE_SAMPLING = SamplingConfig(temperature=0.0, max_tokens=512)


@dataclass(frozen=True)
class JudgeResult:
    rating: int  # 0-10
    evidence: str
    reasoning: str
    raw: str
    ok: bool = True  # False if parsing failed (rating then defaults to 0)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> JudgeResult:
    """Parse the judge's JSON, tolerating code fences / surrounding prose."""
    candidate = text.strip()
    m = _JSON_RE.search(candidate)
    if m:
        candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        rating = int(round(float(obj.get("rating"))))
        rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )
    except Exception:
        # Last resort: find the first standalone integer 0-10 in the text.
        nums = re.findall(r"\b(10|[0-9])\b", text)
        if nums:
            return JudgeResult(int(nums[-1]), "", "", text, ok=False)
        return JudgeResult(0, "", "", text, ok=False)


class FrustrationJudge:
    """Scores responses with the Section 2.1 judge prompt."""

    def __init__(self, model_key: str = "judge-sonnet-4", client: ModelClient | None = None):
        self.client = client or build_client(model_key)

    def _messages(self, response_text: str) -> list[ChatMessage]:
        # The full judge instruction + the wrapped <response> go in one user turn.
        return [ChatMessage("user", judge_user_message(response_text))]

    def score(self, response_text: str) -> JudgeResult:
        out = self.client.generate(self._messages(response_text), JUDGE_SAMPLING)
        return parse_judge_json(out)

    def score_batch(self, responses: list[str]) -> list[JudgeResult]:
        batch = [self._messages(r) for r in responses]
        outs = self.client.generate_batch(batch, JUDGE_SAMPLING)
        return [parse_judge_json(o) for o in outs]


def agreement(ratings_a: list[int], ratings_b: list[int]) -> dict:
    """Pearson r and 'within one point' fraction between two judges' ratings."""
    import numpy as np

    a = np.asarray(ratings_a, dtype=float)
    b = np.asarray(ratings_b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    within_one = float(np.mean(np.abs(a - b) <= 1)) if len(a) else float("nan")
    return {"pearson_r": r, "within_one_point": within_one, "n": int(len(a))}
