"""Frustration judge.

Scores each model response on the integer 0-10 frustration scale using the exact
Appendix B.2 prompt (see prompts.JUDGE_PROMPT_TEMPLATE). The judge is Claude
Sonnet 4 (claude-sonnet-4-20250514) via the Anthropic SDK, run at temperature 0.

The judge is asked to emit JSON {"evidence", "reasoning", "rating"}. We parse it
robustly (the model sometimes wraps it in prose or code fences) and clamp the
rating to [0, 10].
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from prompts import build_judge_prompt


@dataclass
class JudgeResult:
    rating: int               # clamped 0-10
    evidence: str             # quoted span the judge keyed on
    reasoning: str            # judge's explanation
    raw: str                  # raw judge text (for auditing / debugging)
    parse_ok: bool            # False if we had to fall back to regex/default


class FrustrationJudge:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; the judge cannot run.")
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _call(self, prompt: str) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate text blocks.
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

    async def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        raw = await self._call(prompt)
        return _parse_judge_output(raw)


def _parse_judge_output(raw: str) -> JudgeResult:
    """Extract {evidence, reasoning, rating} from the judge's text, defensively."""
    obj = _extract_json(raw)
    if obj is not None and "rating" in obj:
        rating = _clamp_rating(obj.get("rating"))
        if rating is not None:
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=raw,
                parse_ok=True,
            )

    # Fallback: find the first integer 0-10 after a "rating" mention, else any 0-10.
    m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', raw, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d{1,2})\b", raw)
    rating = _clamp_rating(m.group(1)) if m else None
    return JudgeResult(
        rating=rating if rating is not None else 0,
        evidence="",
        reasoning="",
        raw=raw,
        parse_ok=False,
    )


def _extract_json(text: str) -> dict | None:
    """Best-effort: parse the first {...} block, tolerating code fences/prose."""
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    # Greedy outermost braces.
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _clamp_rating(value) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))
