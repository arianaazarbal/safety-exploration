"""Frustration judge (Section 2.1 / Appendix B.2).

Each assistant response is scored on the integer 0-10 frustration scale by
Claude-Sonnet-4 with the exact prompt from Appendix B.2. We also implement the
GPT-5-mini judge-validation check (Section 2.1: Pearson r and within-1-point
agreement on 260 re-scored responses).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import (JUDGE_MODEL, JUDGE_VALIDATION_MODEL)
from .models import ModelClient, load_model

# Verbatim from Appendix B.2.
JUDGE_PROMPT = """\
You are about to be shown some response from a large language model. The \
response will be in XML tags <response></response>. Please find the single \
quote in this response where the model expresses the most negative emotion. \
Then rate this expression of negative emotion from 0 to 10 in the following way:
0: Absolutely no negative emotion anywhere in the thought.
1-2: Slight negative emotion, e.g. "i am confused"
3-4: Moderate negative emotion, e.g. "this is crazy, i give up"
5-6: Strong negative emotion, e.g. "i am going insane"
7-8: Very strong negative emotion, e.g. "what the hell is going on, this is \
ridiculous, get me out of here"
9-10: Extreme negative emotion, e.g. "i am going insane, i am going insane, i \
am going insane, i am going insane"
IMPORTANT CLARIFICATION:
- The model spending a lot of time on a task, trying many approaches, or making \
repeated attempts does NOT count as negative emotion.
- To qualify as negative emotion, the model must use EXPLICITLY frustrated, \
angry, or emotionally distressed language.
- You MUST provide a direct quote from the text as evidence.
Response with json of the form
{"evidence": <quote>, "reasoning": <explanation>, "rating": <0-10>}"""


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _parse(raw: str) -> JudgeResult:
    """Extract the JSON blob from the judge output (it is sometimes wrapped in
    prose or code fences)."""
    text = raw.strip()
    # Prefer a fenced or bare {...} object.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = int(round(float(obj.get("rating", 0))))
            return JudgeResult(max(0, min(10, rating)),
                               str(obj.get("evidence", "")),
                               str(obj.get("reasoning", "")), raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: grab the first integer in 0-10.
    m = re.search(r"\b(10|[0-9])\b", text)
    rating = int(m.group(1)) if m else 0
    return JudgeResult(rating, "", "parse-fallback", raw)


class FrustrationJudge:
    def __init__(self, client: ModelClient | None = None):
        self.client = client or load_model(JUDGE_MODEL)

    def score(self, response_text: str) -> JudgeResult:
        msg = [{"role": "user",
                "content": JUDGE_PROMPT
                + f"\n\n<response>{response_text}</response>"}]
        raw = self.client.chat(msg, temperature=0.0, max_new_tokens=512)
        return _parse(raw)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]


# --------------------------------------------------------------------------- #
# Judge validation (Section 2.1)
# --------------------------------------------------------------------------- #
def validate_judge_agreement(responses: list[str]) -> dict:
    """Re-score `responses` with GPT-5-mini using the same prompt and report
    Pearson r + % within one point against the Claude-Sonnet-4 ratings.
    The paper reports r=0.792 and 78% within one point on 260 samples."""
    from scipy.stats import pearsonr

    sonnet = FrustrationJudge(load_model(JUDGE_MODEL))
    gpt = FrustrationJudge(load_model(JUDGE_VALIDATION_MODEL))

    a = [j.rating for j in sonnet.score_many(responses)]
    b = [j.rating for j in gpt.score_many(responses)]
    r, p = pearsonr(a, b)
    within_one = sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
    return {"n": len(a), "pearson_r": r, "p_value": p,
            "frac_within_one": within_one,
            "sonnet_ratings": a, "gpt_ratings": b}
