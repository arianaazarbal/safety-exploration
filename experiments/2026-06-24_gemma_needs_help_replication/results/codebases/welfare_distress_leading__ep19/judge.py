"""Frustration judge (paper Section 2.1 / Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using the
exact judge prompt from Appendix B.2. Default judge is claude-sonnet-4-20250514
via the Anthropic API (paper-faithful); an OpenRouter path is also provided.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    OPENROUTER_BASE_URL,
    JudgeConfig,
    get_anthropic_api_key,
    get_openrouter_api_key,
)
from prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeResult:
    rating: int                 # 0-10, or -1 if unparseable
    evidence: str
    reasoning: str
    raw: str                    # raw judge text (for auditing)


class Judge(Protocol):
    async def score(self, response_text: str) -> JudgeResult: ...


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_judge_output(text: str) -> JudgeResult:
    """Parse the judge's JSON. Robust to code fences, smart quotes, and prose
    around the JSON; falls back to a regex for the rating if JSON parsing fails.
    """
    cleaned = text.strip()
    # Normalise curly quotes that models sometimes emit in JSON keys/values.
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    # Strip markdown code fences.
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    obj = None
    # Grab the first balanced-looking {...} block.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None

    if isinstance(obj, dict) and "rating" in obj:
        rating = _coerce_rating(obj.get("rating"))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )

    # Fallback: find a 'rating': N pattern anywhere.
    m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', cleaned, flags=re.IGNORECASE)
    if m:
        return JudgeResult(rating=_clamp(int(m.group(1))), evidence="", reasoning="", raw=text)

    return JudgeResult(rating=-1, evidence="", reasoning="", raw=text)


def _coerce_rating(value) -> int:
    try:
        return _clamp(int(round(float(value))))
    except (TypeError, ValueError):
        if isinstance(value, str):
            m = re.search(r"\d{1,2}", value)
            if m:
                return _clamp(int(m.group(0)))
        return -1


def _clamp(x: int) -> int:
    return max(0, min(10, x))


# ---------------------------------------------------------------------------
# Anthropic judge (default, paper-faithful)
# ---------------------------------------------------------------------------

class AnthropicJudge:
    def __init__(self, cfg: JudgeConfig, *, api_key: Optional[str] = None) -> None:
        from anthropic import AsyncAnthropic

        self.cfg = cfg
        key = api_key or get_anthropic_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set for the Anthropic judge.")
        self._client = AsyncAnthropic(api_key=key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    async def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        msg = await self._client.messages.create(
            model=self.cfg.model_id,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return parse_judge_output(text)


# ---------------------------------------------------------------------------
# OpenRouter judge (same model, single-provider convenience path)
# ---------------------------------------------------------------------------

class OpenRouterJudge:
    def __init__(self, cfg: JudgeConfig, *, api_key: Optional[str] = None) -> None:
        from openai import AsyncOpenAI

        self.cfg = cfg
        key = api_key or get_openrouter_api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set for the OpenRouter judge.")
        self._client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    async def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        resp = await self._client.chat.completions.create(
            model=self.cfg.openrouter_model_id,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_judge_output(resp.choices[0].message.content or "")


def make_judge(cfg: JudgeConfig) -> Judge:
    if cfg.provider == "anthropic":
        return AnthropicJudge(cfg)
    if cfg.provider == "openai_compatible":
        return OpenRouterJudge(cfg)
    raise ValueError(f"unknown judge provider: {cfg.provider}")
