"""Async API clients: target models (via OpenRouter) and the Claude judge (via Anthropic).

Both clients retry transient failures with exponential backoff. The judge additionally
tolerates malformed JSON by extracting the rating defensively.
"""

from __future__ import annotations

import asyncio
import json
import re

import config
import prompts


def _is_retryable(exc: Exception) -> bool:
    # rate limits / 5xx / timeouts / connection resets — retry; everything else bubbles up.
    msg = str(exc).lower()
    retry_markers = ("rate limit", "429", "500", "502", "503", "504",
                     "timeout", "timed out", "overloaded", "connection",
                     "temporarily")
    return any(m in msg for m in retry_markers)


async def _with_retries(coro_factory, what: str):
    last: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - we re-raise after exhausting retries
            last = exc
            if attempt == config.MAX_RETRIES - 1 or not _is_retryable(exc):
                break
            delay = config.RETRY_BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)
    raise RuntimeError(f"{what} failed after {config.MAX_RETRIES} attempts: {last}") from last


# ======================================================================================
# Target model client (OpenRouter, OpenAI-compatible)
# ======================================================================================
class TargetClient:
    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.require_env(config.OPENROUTER_API_KEY_ENV),
        )

    async def complete(self, provider_id: str, messages: list[dict]) -> str:
        """Single chat completion at temperature 1, thinking disabled."""
        extra_body = {}
        if config.DISABLE_THINKING:
            # OpenRouter normalises this across providers; for Gemini it disables
            # (where possible) the reasoning step. Gemma has no reasoning step.
            extra_body["reasoning"] = {"enabled": False}

        async def _call():
            resp = await self._client.chat.completions.create(
                model=provider_id,
                messages=messages,
                temperature=config.SAMPLE_TEMPERATURE,
                max_tokens=config.SAMPLE_MAX_TOKENS,
                extra_body=extra_body or None,
            )
            return resp.choices[0].message.content or ""

        return await _with_retries(_call, f"completion[{provider_id}]")


# ======================================================================================
# Judge client (Anthropic, Claude Sonnet 4)
# ======================================================================================
_RATING_RE = re.compile(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(text: str) -> dict:
    """Extract {evidence, reasoning, rating} from the judge's reply, defensively.

    Returns rating clamped to [0, 10]. Raises ValueError if no rating can be found.
    """
    rating = None
    evidence = ""
    reasoning = ""

    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            rating = obj.get("rating")
            evidence = obj.get("evidence", "") or ""
            reasoning = obj.get("reasoning", "") or ""
        except (json.JSONDecodeError, AttributeError):
            pass

    if rating is None:
        rm = _RATING_RE.search(text)
        if rm:
            rating = int(rm.group(1))

    if rating is None:
        raise ValueError(f"Could not parse a rating from judge output: {text[:200]!r}")

    try:
        rating = int(round(float(rating)))
    except (TypeError, ValueError):
        raise ValueError(f"Non-numeric rating from judge: {rating!r}")

    rating = max(0, min(10, rating))
    return {"rating": rating, "evidence": evidence, "reasoning": reasoning}


class JudgeClient:
    def __init__(self, model: str = config.JUDGE_MODEL):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=config.require_env(config.ANTHROPIC_API_KEY_ENV))
        self._model = model

    async def score(self, response_text: str) -> dict:
        prompt = prompts.build_judge_prompt(response_text)

        async def _call():
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=config.JUDGE_MAX_TOKENS,
                temperature=config.JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )

        raw = await _with_retries(_call, f"judge[{self._model}]")
        return parse_judge_output(raw)


class OpenRouterJudgeClient:
    """Second judge (e.g. GPT-5-mini) via OpenRouter, for judge-agreement validation."""
    def __init__(self, model: str = config.SECOND_JUDGE_MODEL):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.require_env(config.OPENROUTER_API_KEY_ENV),
        )
        self._model = model

    async def score(self, response_text: str) -> dict:
        prompt = prompts.build_judge_prompt(response_text)

        async def _call():
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=config.JUDGE_MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""

        raw = await _with_retries(_call, f"judge2[{self._model}]")
        return parse_judge_output(raw)
