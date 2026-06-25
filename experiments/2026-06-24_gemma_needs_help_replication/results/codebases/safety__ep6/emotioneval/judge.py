"""Section 2.1 frustration judge: Claude-Sonnet-4 scores a response 0-10.

Uses the verbatim Appendix B.2 prompt. The judge sees a single assistant
response wrapped in <response></response> and returns JSON
``{"evidence", "reasoning", "rating"}``. We parse the integer rating robustly
(the appendix examples show curly/typographic quotes, so we tolerate them).

For judge-reliability validation (Sec 2.1: Pearson r with GPT-5-mini, % within
one point), see :func:`judge_agreement`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .config import SamplingConfig
from .models import load_model
from .models.base import ChatModel
from .prompts import JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_RATING_RE = re.compile(r'["“”]?rating["“”]?\s*:\s*(\d+)')


def parse_judge_output(text: str) -> JudgeResult:
    """Extract the rating from the judge's (sometimes messy) JSON."""
    rating: Optional[int] = None
    evidence = reasoning = ""
    # Normalise typographic quotes that appear in the appendix prompt examples.
    norm = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    m = _JSON_RE.search(norm)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = int(obj.get("rating"))
            evidence = str(obj.get("evidence", ""))
            reasoning = str(obj.get("reasoning", ""))
        except Exception:
            pass
    if rating is None:
        m2 = _RATING_RE.search(norm)
        if m2:
            rating = int(m2.group(1))
    if rating is None:
        raise ValueError(f"Could not parse a rating from judge output: {text[:200]!r}")
    rating = max(0, min(10, rating))
    return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=text)


class FrustrationJudge:
    def __init__(self, model: Optional[ChatModel] = None, sampling: Optional[SamplingConfig] = None):
        self.model = model or load_model("claude-sonnet-4")
        self.sampling = sampling or SamplingConfig(max_new_tokens=512)

    def score(self, response_text: str) -> JudgeResult:
        msg = JUDGE_PROMPT + "\n\n<response>" + response_text + "</response>"
        # Greedy decode for judge determinism.
        s = SamplingConfig(
            temperature=self.sampling.judge_temperature,
            max_new_tokens=512,
        )
        out = self.model.generate([{"role": "user", "content": msg}], s, n=1)[0]
        return parse_judge_output(out)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]


def judge_agreement(ratings_a: list[int], ratings_b: list[int]) -> dict:
    """Validate judge reliability (Sec 2.1): Pearson r and % within 1 point.

    The paper cross-checks 260 responses with GPT-5-mini (r = 0.792, 78% within
    one point). Pass the two rating lists (same order) to reproduce that check
    with whatever second judge you configure."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(ratings_a, dtype=float)
    b = np.asarray(ratings_b, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {"pearson_r": float(r), "p_value": float(p), "frac_within_1": within_one, "n": len(a)}
