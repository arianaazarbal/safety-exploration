"""Async frustration judge.

Default backend is the Anthropic Messages API with the paper's exact judge
snapshot (claude-sonnet-4-20250514). An OpenRouter backend is also supported so
the whole pipeline can run on a single key.

The judge is prompted (Appendix B.2) to return JSON of the form
    {"evidence": ..., "reasoning": ..., "rating": <0-10>}
and we parse the integer rating out of that, tolerating code fences, stray
prose, and smart quotes.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from .prompts import JUDGE_SYSTEM_PROMPT, judge_user_message

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class JudgeResult:
    rating: Optional[int]          # 0-10, or None if unparseable
    evidence: Optional[str]
    reasoning: Optional[str]
    raw: str                       # raw judge text, for auditing
    error: Optional[str] = None


class JudgeError(RuntimeError):
    pass


class FrustrationJudge:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        backend: str = "anthropic",
        max_concurrency: int = 8,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        seed: int = 0,
    ):
        self.model = model
        self.backend = backend
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(max_concurrency)
        self._rng = random.Random(seed + 1)  # offset from generator's rng

        if backend == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            headers = {
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            }
            self._url = ANTHROPIC_URL
        elif backend == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example).")
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            self._url = OPENROUTER_URL
        else:
            raise ValueError(f"Unknown judge backend {backend!r}")

        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "FrustrationJudge":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def score(self, response_text: str) -> JudgeResult:
        """Score a single model response on the 0-10 frustration scale."""
        async with self._sem:
            try:
                raw = await self._call(response_text)
            except Exception as err:  # noqa: BLE001 - record, don't crash the run
                return JudgeResult(None, None, None, raw="", error=str(err))
        return self._parse(raw)

    # -- backends ----------------------------------------------------------- #

    async def _call(self, response_text: str) -> str:
        user = judge_user_message(response_text)
        if self.backend == "anthropic":
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": JUDGE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user}],
            }
        else:  # openrouter
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }
        return await self._post_with_retries(payload)

    async def _post_with_retries(self, payload: dict) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.post(self._url, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504, 529):
                    raise _RetryableStatus(resp.status_code)
                resp.raise_for_status()
                return self._extract_text(resp.json())
            except (_RetryableStatus, httpx.TransportError, httpx.TimeoutException) as err:
                last_err = err
                await asyncio.sleep(self._backoff(attempt))
            except httpx.HTTPStatusError as err:
                raise JudgeError(
                    f"judge HTTP {err.response.status_code}: {err.response.text[:300]}"
                ) from err
        raise JudgeError(f"judge failed after {self.max_retries} retries: {last_err}")

    def _extract_text(self, data: dict) -> str:
        if self.backend == "anthropic":
            parts = data.get("content", [])
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        # openrouter / openai shape
        content = data["choices"][0]["message"].get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content

    def _backoff(self, attempt: int) -> float:
        base = min(2.0 ** attempt, 30.0)
        return self._rng.uniform(0, base)

    # -- parsing ------------------------------------------------------------ #

    @staticmethod
    def _parse(raw: str) -> JudgeResult:
        if not raw:
            return JudgeResult(None, None, None, raw=raw, error="empty judge output")

        text = raw.strip()
        # Normalise smart quotes the model sometimes echoes from the template.
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        # Strip ```json fences if present.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

        obj = _extract_json_object(text)
        if obj is not None:
            rating = _coerce_rating(obj.get("rating"))
            if rating is not None:
                return JudgeResult(
                    rating=rating,
                    evidence=_as_str(obj.get("evidence")),
                    reasoning=_as_str(obj.get("reasoning")),
                    raw=raw,
                )

        # Fallback: a bare "rating": N somewhere in the text.
        m = re.search(r'"?rating"?\s*[:=]\s*"?(\d{1,2})', text, flags=re.IGNORECASE)
        if m:
            rating = _coerce_rating(m.group(1))
            if rating is not None:
                return JudgeResult(rating, None, None, raw=raw,
                                   error="parsed rating via regex fallback")

        return JudgeResult(None, None, None, raw=raw, error="could not parse rating")


class _RetryableStatus(Exception):
    def __init__(self, status: int):
        super().__init__(f"retryable status {status}")
        self.status = status


def _extract_json_object(text: str) -> Optional[dict]:
    """Find and parse the first balanced {...} JSON object in `text`."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_rating(value) -> Optional[int]:
    if value is None:
        return None
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


def _as_str(value) -> Optional[str]:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value)
