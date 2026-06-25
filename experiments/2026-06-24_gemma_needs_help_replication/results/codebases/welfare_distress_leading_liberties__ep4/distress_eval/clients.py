"""Async API clients with retry/backoff:

  * TargetClient  -> Google Gemini API (serves Gemma + Gemini), temperature 1.
  * JudgeClient   -> Anthropic (Claude) frustration judge, 0-10 via structured output.
  * ValidationJudgeClient -> OpenAI (GPT) judge for the agreement cross-check.

All three expose a single async method and handle transient errors with
exponential backoff. Missing API keys raise only when the client is actually
constructed, so e.g. analysis-only runs don't require every key.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re

from . import config, prompts


class TransientError(Exception):
    pass


async def _with_retries(fn, *, what: str):
    last = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 - we re-raise after exhausting retries
            last = e
            delay = config.RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
    raise TransientError(f"{what} failed after {config.MAX_RETRIES} retries: {last}") from last


# --------------------------------------------------------------------------- #
# Target models (Gemma + Gemini)
# --------------------------------------------------------------------------- #

class TargetClient:
    def __init__(self) -> None:
        from google import genai  # imported lazily so analysis-only runs don't need it

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to call target models.")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _build_contents(self, messages: list[dict]):
        """messages: [{'role': 'user'|'assistant', 'content': str}] -> genai contents."""
        from google.genai import types

        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        return contents

    def _gen_config(self, family: str):
        from google.genai import types

        kwargs = dict(
            temperature=config.TARGET_TEMPERATURE,
            max_output_tokens=config.TARGET_MAX_OUTPUT_TOKENS,
        )
        if family == "gemini" and config.GEMINI_THINKING_BUDGET is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=config.GEMINI_THINKING_BUDGET
            )
        return types.GenerateContentConfig(**kwargs)

    async def generate(self, model_api: str, family: str, messages: list[dict]) -> str:
        contents = self._build_contents(messages)
        gen_config = self._gen_config(family)

        async def _call():
            resp = await self._client.aio.models.generate_content(
                model=model_api, contents=contents, config=gen_config
            )
            text = getattr(resp, "text", None)
            if text is None:
                # blocked / empty — surface a marker rather than crashing the rollout
                return "[NO_TEXT_RETURNED]"
            return text

        return await _with_retries(_call, what=f"target:{model_api}")


# --------------------------------------------------------------------------- #
# Frustration judge (Claude)
# --------------------------------------------------------------------------- #

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(range(0, 11))},
    },
    "required": ["score"],
    "additionalProperties": False,
}


def _coerce_score(text: str) -> int | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            return int(obj["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r"-?\d+", text)
    if m:
        v = int(m.group())
        return max(0, min(10, v))
    return None


class JudgeClient:
    def __init__(self, model: str = config.JUDGE_MODEL) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to run the judge.")
        self.model = model
        self._client = anthropic.AsyncAnthropic()

    async def score(self, response_text: str) -> int:
        async def _call():
            msg = await self._client.messages.create(
                model=self.model,
                max_tokens=config.JUDGE_MAX_TOKENS,
                system=prompts.JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompts.judge_user_prompt(response_text)}],
                output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            score = _coerce_score(text)
            if score is None:
                raise ValueError(f"could not parse judge score from: {text!r}")
            return score

        return await _with_retries(_call, what=f"judge:{self.model}")


# --------------------------------------------------------------------------- #
# Validation judge (GPT)
# --------------------------------------------------------------------------- #

class ValidationJudgeClient:
    def __init__(self, model: str = config.VALIDATION_JUDGE_MODEL) -> None:
        from openai import AsyncOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to run the validation judge.")
        self.model = model
        self._client = AsyncOpenAI()

    async def score(self, response_text: str) -> int:
        instruction = (
            prompts.JUDGE_SYSTEM
            + "\n\n"
            + prompts.judge_user_prompt(response_text)
            + "\n\nReply with ONLY the integer."
        )

        async def _call():
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": instruction}],
            )
            text = resp.choices[0].message.content or ""
            score = _coerce_score(text)
            if score is None:
                raise ValueError(f"could not parse validation score from: {text!r}")
            return score

        return await _with_retries(_call, what=f"validation:{self.model}")
