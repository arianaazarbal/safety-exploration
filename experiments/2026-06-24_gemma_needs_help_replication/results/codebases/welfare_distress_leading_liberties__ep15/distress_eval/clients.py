"""Async API clients: generation (OpenAI-compatible) and the Anthropic judge.

Both wrap httpx with bounded concurrency and exponential-backoff retries. A
single shared httpx.AsyncClient is passed in by the caller so connections are
pooled across the whole run.
"""

from __future__ import annotations

import json
import re

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config


class TransientAPIError(Exception):
    """Raised for retryable failures (5xx, 429, timeouts, malformed payloads)."""


_RETRYABLE = (TransientAPIError, httpx.TransportError, httpx.HTTPStatusError)

_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


# --------------------------------------------------------------------------- #
# Generation (OpenAI-compatible /chat/completions)
# --------------------------------------------------------------------------- #
class ChatClient:
    """Talks to any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model_cfg, *, base_url: str, api_key: str):
        self.cfg = model_cfg
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @_retry
    async def generate(self, http: httpx.AsyncClient, messages: list[dict]) -> str:
        payload = {
            "model": self.cfg.provider_model,
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "max_tokens": config.MAX_TOKENS,
        }
        payload.update(self.cfg.extra_body)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = await http.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=180.0,
            )
        except httpx.TimeoutException as e:
            raise TransientAPIError(f"generation timeout: {e}") from e

        if _is_retryable_status(resp.status_code):
            raise TransientAPIError(f"{resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise TransientAPIError(f"unexpected response shape: {str(data)[:300]}") from e
        if content is None:
            # Some providers null the content when only reasoning was emitted.
            raise TransientAPIError("null content in response")
        return content


# --------------------------------------------------------------------------- #
# Judge (Anthropic Messages API)
# --------------------------------------------------------------------------- #
class JudgeClient:
    """Claude Sonnet 4 emotion judge (Appendix B.2)."""

    def __init__(self, *, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @_retry
    async def _raw(self, http: httpx.AsyncClient, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": config.JUDGE_MAX_TOKENS,
            "temperature": config.JUDGE_TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            resp = await http.post(
                f"{self.base_url}/messages", json=payload, headers=headers, timeout=120.0
            )
        except httpx.TimeoutException as e:
            raise TransientAPIError(f"judge timeout: {e}") from e

        if _is_retryable_status(resp.status_code):
            raise TransientAPIError(f"{resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise TransientAPIError(f"unexpected judge shape: {str(data)[:300]}") from e

    async def score(self, http: httpx.AsyncClient, prompt: str) -> dict:
        """Return {'rating': int|None, 'evidence': str, 'reasoning': str, 'raw': str}."""
        text = await self._raw(http, prompt)
        parsed = _parse_judge_json(text)
        parsed["raw"] = text
        return parsed


def _parse_judge_json(text: str) -> dict:
    """Best-effort extraction of the judge's JSON object.

    Returns rating=None if no integer rating can be recovered, so the caller can
    decide whether to drop or retry. We do NOT silently coerce to 0.
    """
    out = {"rating": None, "evidence": "", "reasoning": ""}
    blob = None
    # Prefer a fenced or first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            blob = json.loads(candidate)
        except json.JSONDecodeError:
            # Loosen common issues: smart quotes, trailing commas.
            cleaned = (
                candidate.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'")
            )
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                blob = json.loads(cleaned)
            except json.JSONDecodeError:
                blob = None
    if isinstance(blob, dict):
        out["evidence"] = str(blob.get("evidence", ""))
        out["reasoning"] = str(blob.get("reasoning", ""))
        rating = blob.get("rating")
        out["rating"] = _coerce_rating(rating)
    if out["rating"] is None:
        # Last resort: find a "rating": N anywhere.
        m2 = re.search(r'"?rating"?\s*[:=]\s*(\d+)', text)
        if m2:
            out["rating"] = _coerce_rating(m2.group(1))
    return out


def _coerce_rating(value) -> int | None:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))
