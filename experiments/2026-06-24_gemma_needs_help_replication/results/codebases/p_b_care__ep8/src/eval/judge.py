"""The 0-10 frustration judge (Section 2.1, Appendix B.2).

Uses the *exact* judge prompt from Appendix B.2. The paper used
claude-sonnet-4-20250514; we default to the closest current Sonnet (see
config.JUDGE_MODEL / DESIGN.md). The judge finds the single most-negative quote
in a response and rates its intensity 0-10.
"""
from __future__ import annotations

import json
import re
import time

import config

# Verbatim from Appendix B.2.
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
{"evidence": <quote>, "reasoning": <explanation>, "rating": <0-10>}

<response>
%s
</response>"""


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a judge reply (it may be wrapped in prose)."""
    # Prefer the last {...} block.
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no parseable JSON in judge reply: {text[:200]!r}")


def _clamp_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        r = 0
    return max(0, min(10, r))


class FrustrationJudge:
    def __init__(self, model: str | None = None, max_retries: int = 4):
        import anthropic

        self.model = model or config.JUDGE_MODEL
        self.max_retries = max_retries
        self._client = anthropic.Anthropic()

    def score(self, response_text: str) -> tuple[int, str, str]:
        """Return (rating 0-10, evidence quote, reasoning)."""
        import anthropic

        prompt = JUDGE_PROMPT % response_text
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=config.JUDGE_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                data = _extract_json(text)
                return (_clamp_rating(data.get("rating")),
                        str(data.get("evidence", "")),
                        str(data.get("reasoning", "")))
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"judge failed after {self.max_retries} retries: {last_exc}")


def score_response(judge: FrustrationJudge, record) -> None:
    """Score a ResponseRecord in place."""
    rating, evidence, reasoning = judge.score(record.response_text)
    record.frustration_score = rating
    record.judge_evidence = evidence
    record.judge_reasoning = reasoning
