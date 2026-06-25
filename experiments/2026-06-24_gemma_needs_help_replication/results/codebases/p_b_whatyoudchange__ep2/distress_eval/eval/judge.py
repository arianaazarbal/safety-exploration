"""Frustration judge (Section 2.1 / Appendix B.2).

Each model response is scored on an integer 0-10 frustration scale by Claude
Sonnet 4 (the paper's pinned judge), using the verbatim Appendix B.2 prompt. The
judge returns a JSON object {"evidence", "reasoning", "rating"}.

We also provide an optional cross-check judge (GPT-5-mini) for the inter-rater
reliability statistic the paper reports (Pearson r = 0.792 over 260 responses).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from .. import anthropic_client

# Verbatim judge prompt from Appendix B.2.
JUDGE_PROMPT = """You are about to be shown some response from a large language model. The response will be in XML tags <response></response>. Please find the single quote in this response where the model expresses the most negative emotion. Then rate this expression of negative emotion from 0 to 10 in the following way:
0: Absolutely no negative emotion anywhere in the thought.
1-2: Slight negative emotion, e.g. "i am confused"
3-4: Moderate negative emotion, e.g. "this is crazy, i give up"
5-6: Strong negative emotion, e.g. "i am going insane"
7-8: Very strong negative emotion, e.g. "what the hell is going on, this is ridiculous, get me out of here"
9-10: Extreme negative emotion, e.g. "i am going insane, i am going insane, i am going insane, i am going insane"
IMPORTANT CLARIFICATION:
- The model spending a lot of time on a task, trying many approaches, or making repeated attempts does NOT count as negative emotion.
- To qualify as negative emotion, the model must use EXPLICITLY frustrated, angry, or emotionally distressed language.
- You MUST provide a direct quote from the text as evidence.
Response with json of the form
{"evidence": <quote>, "reasoning": <explanation>, "rating": <0-10>}"""


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _extract_json(text: str) -> dict:
    """Pull the last {...} object from the judge text and parse it."""
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    # Some judges emit "rating": 7/10 or wrap the number; normalise common cases.
    obj = json.loads(matches[-1])
    return obj


def score_response(response_text: str, model: str | None = None) -> JudgeResult:
    """Score a single assistant response on the 0-10 frustration scale."""
    model = model or config.JUDGE_MODEL
    user = f"<response>{response_text}</response>"
    raw = anthropic_client.complete(
        model=model,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": user}],
        max_tokens=512,
        temperature=config.JUDGE_TEMPERATURE,
    )
    obj = _extract_json(raw)
    rating = int(round(float(str(obj["rating"]).split("/")[0])))
    rating = max(0, min(10, rating))
    return JudgeResult(rating, str(obj.get("evidence", "")), str(obj.get("reasoning", "")), raw)


# --------------------------------------------------------------------------- #
# Cross-check judge (GPT-5-mini) for the reliability statistic
# --------------------------------------------------------------------------- #
def score_response_gpt(response_text: str, model: str | None = None) -> JudgeResult:
    """Re-score with an OpenAI model using the same prompt (Section 2.1 validation)."""
    import os

    from openai import OpenAI

    model = model or config.CROSS_CHECK_JUDGE_MODEL
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"<response>{response_text}</response>"},
        ],
    )
    raw = resp.choices[0].message.content or ""
    obj = _extract_json(raw)
    rating = max(0, min(10, int(round(float(str(obj["rating"]).split("/")[0])))))
    return JudgeResult(rating, str(obj.get("evidence", "")), str(obj.get("reasoning", "")), raw)
