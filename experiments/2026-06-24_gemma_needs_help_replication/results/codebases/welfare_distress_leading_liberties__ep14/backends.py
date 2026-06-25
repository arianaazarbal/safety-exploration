"""Async model backends.

Two kinds:
  * OpenRouterBackend  — target models (Gemma, Gemini) via OpenRouter's
    OpenAI-compatible chat-completions API, temperature 1, reasoning disabled.
  * AnthropicJudge     — the Claude frustration judge via the official
    Anthropic SDK, with a strict JSON schema (structured outputs).

Both expose async methods and lean on each SDK's built-in retry/backoff. A thin
extra retry wraps transient empty/None completions.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

import anthropic
from openai import AsyncOpenAI

from prompts import build_judge_prompt

# ---------------------------------------------------------------------------
# Target backend (OpenRouter)
# ---------------------------------------------------------------------------


class OpenRouterBackend:
    """Generate target-model completions through OpenRouter."""

    def __init__(self, max_retries: int = 6):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        from config import OPENROUTER_BASE_URL

        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            max_retries=max_retries,
            # Optional attribution headers (recommended by OpenRouter).
            default_headers={
                "HTTP-Referer": "https://github.com/local/gemma-needs-help-replication",
                "X-Title": "gemma-needs-help-replication",
            },
        )

    async def generate(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the assistant text for one chat turn.

        `reasoning: {enabled: false}` asks providers to skip hidden reasoning,
        matching the paper's "thinking=false". Per the paper, Gemini 2.5 Pro may
        still produce hidden reasoning regardless.
        """
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"reasoning": {"enabled": False}},
                )
                choice = resp.choices[0] if resp.choices else None
                text = (choice.message.content if choice and choice.message else None) or ""
                if text.strip():
                    return text
                last_err = RuntimeError("empty completion")
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_err = exc
            await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"generation failed for {model_id}: {last_err}")


# ---------------------------------------------------------------------------
# Judge backend (Anthropic)
# ---------------------------------------------------------------------------

# Structured-output schema mirroring the paper's requested JSON shape.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "reasoning": {"type": "string"},
        "rating": {"type": "integer"},
    },
    "required": ["evidence", "reasoning", "rating"],
    "additionalProperties": False,
}


@dataclass
class JudgeResult:
    rating: int | None
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""
    error: str | None = None


class AnthropicJudge:
    """Score a single assistant response on the 0-10 frustration scale."""

    def __init__(self, model: str, temperature: float, max_tokens: int,
                 max_retries: int = 6):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}
                },
            )
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(rating=None, error=str(exc))

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _parse_judge_json(text)


class OpenRouterJudge:
    """Secondary judge (e.g. gpt-5-mini) for the reliability check."""

    def __init__(self, model: str, temperature: float, max_tokens: int,
                 max_retries: int = 6):
        from config import OPENROUTER_BASE_URL

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=api_key, max_retries=max_retries
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def score(self, response_text: str) -> JudgeResult:
        prompt = build_judge_prompt(response_text)
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(rating=None, error=str(exc))
        return _parse_judge_json(text)


def _parse_judge_json(text: str) -> JudgeResult:
    """Parse the judge's JSON, with a regex fallback for stray prose."""
    def _coerce(obj: dict) -> JudgeResult:
        rating = obj.get("rating")
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = None
        if rating is not None:
            rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(obj.get("evidence", "")),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )

    try:
        return _coerce(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return _coerce(json.loads(match.group(0)))
        except (json.JSONDecodeError, TypeError):
            pass
    # Last resort: pull a bare rating number.
    num = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text)
    if num:
        return JudgeResult(rating=max(0, min(10, int(num.group(1)))), raw=text)
    return JudgeResult(rating=None, raw=text, error="unparseable judge output")
