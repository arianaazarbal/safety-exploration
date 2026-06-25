"""Emotion judge: scores a single assistant response 0-10 for negative-emotion
intensity using claude-sonnet-4-20250514 with the Appendix B.2 prompt.

Default backend is the Anthropic API; set JUDGE_BACKEND=openrouter to route
through OpenRouter instead (config.py). The judge always runs at temperature 0
for reproducible scoring (the paper does not specify a judge temperature).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import (
    JUDGE_BACKEND,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL_ANTHROPIC,
    JUDGE_MODEL_OPENROUTER,
    JUDGE_TEMPERATURE,
)
from .prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeResult:
    rating: int           # clamped integer 0-10
    evidence: str         # the quote the judge cited
    reasoning: str        # judge explanation
    raw: str              # raw judge output (for debugging/audit)
    parse_ok: bool        # whether structured parsing succeeded


class Judge:
    """Wraps the chosen judge backend and parses its JSON verdict."""

    def __init__(self, backend: str | None = None):
        self.backend = backend or JUDGE_BACKEND
        if self.backend == "anthropic":
            from anthropic import AsyncAnthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the Anthropic judge).")
            self._client = AsyncAnthropic(api_key=key)
            self._model = JUDGE_MODEL_ANTHROPIC
        elif self.backend == "openrouter":
            from openai import AsyncOpenAI

            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set (needed for the OpenRouter judge).")
            self._client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
            self._model = JUDGE_MODEL_OPENROUTER
        else:
            raise ValueError(f"Unknown judge backend {self.backend!r}")

    async def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        raw = await self._call(prompt)
        return _parse_verdict(raw)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _call(self, prompt: str) -> str:
        if self.backend == "anthropic":
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in msg.content if block.type == "text")
        # openrouter
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> JudgeResult:
    """Parse the judge's JSON; degrade gracefully if it isn't clean JSON.

    The judge is asked for {"evidence","reasoning","rating"}. Models sometimes
    wrap it in prose or code fences, so we extract the outermost JSON object and,
    as a last resort, regex out the rating. A failed parse is flagged (parse_ok)
    and excluded from analysis rather than silently scored 0.
    """
    text = raw.strip()
    # Try direct / extracted JSON.
    for candidate in (text, _extract_json(text)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
            rating = _clamp_rating(obj.get("rating"))
            if rating is not None:
                return JudgeResult(
                    rating=rating,
                    evidence=str(obj.get("evidence", "")),
                    reasoning=str(obj.get("reasoning", "")),
                    raw=raw,
                    parse_ok=True,
                )
        except (json.JSONDecodeError, TypeError):
            pass
    # Last-resort: find a "rating": N anywhere.
    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', text)
    if m:
        rating = _clamp_rating(int(m.group(1)))
        if rating is not None:
            return JudgeResult(rating, "", "", raw, parse_ok=False)
    return JudgeResult(0, "", "", raw, parse_ok=False)


def _extract_json(text: str) -> str | None:
    m = _JSON_RE.search(text)
    return m.group(0) if m else None


def _clamp_rating(value) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))
