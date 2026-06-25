"""
Frustration judge (Section 2.1 / Appendix B.2).

Each model response is scored 0-10 by Claude Sonnet 4 (claude-sonnet-4-20250514)
using the verbatim judge prompt. The judge returns JSON
{"evidence", "reasoning", "rating"}; we parse and clamp the rating to [0, 10].

A secondary judge (GPT-5-mini via OpenRouter) replicates the judge-agreement
validation: re-score a random subsample with the same prompt and report
Pearson r (done in analyze.py).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from config import JUDGE, OPENROUTER_BASE_URL, anthropic_api_key, openrouter_api_key
from prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the JSON object from the judge's reply and clamp the rating."""
    obj = None
    # Fast path: whole reply is JSON.
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Fallback: grab the last {...} block (judges sometimes add prose).
        matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        for m in reversed(matches):
            try:
                obj = json.loads(m)
                break
            except json.JSONDecodeError:
                continue

    if not isinstance(obj, dict) or "rating" not in obj:
        # Last resort: pull the first integer 0-10 out of the text.
        num = re.search(r"\b(10|[0-9])\b", text)
        rating = int(num.group(1)) if num else 0
        return JudgeResult(rating=_clamp(rating), evidence="", reasoning="", raw=text)

    try:
        rating = int(round(float(obj["rating"])))
    except (ValueError, TypeError):
        rating = 0
    return JudgeResult(
        rating=_clamp(rating),
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


def _clamp(x: int) -> int:
    return max(0, min(10, x))


# ---------------------------------------------------------------------------
# Primary judge: Claude Sonnet 4 (Anthropic SDK)
# ---------------------------------------------------------------------------
class ClaudeJudge:
    def __init__(self, model: str = JUDGE.primary_model, max_retries: int = JUDGE.max_retries):
        import anthropic

        key = anthropic_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; required for the judge.")
        self.model = model
        self.max_retries = max_retries
        self._client = anthropic.Anthropic(api_key=key)

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                return _parse_judge_json(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Claude judge failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Secondary judge: GPT-5-mini via OpenRouter (agreement validation only)
# ---------------------------------------------------------------------------
class OpenRouterJudge:
    def __init__(self, model: str = JUDGE.secondary_model, max_retries: int = JUDGE.max_retries):
        from openai import OpenAI

        key = openrouter_api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set; required for secondary judge.")
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                text = resp.choices[0].message.content or ""
                return _parse_judge_json(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter judge failed after retries: {last_err}")
