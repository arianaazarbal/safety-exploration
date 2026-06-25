"""Async API clients for generation models and the emotion judge.

Two provider families are supported out of the box:

  generation:
    - openrouter   : OpenAI-compatible /chat/completions (Gemma + Gemini)
    - local_vllm   : same wire format, pointed at a self-hosted vLLM server
                     (for running Gemma on local GPUs, faithful to the paper)
  judge:
    - anthropic    : Anthropic /v1/messages (claude-sonnet-4-20250514)
    - openrouter   : Claude Sonnet 4 routed through OpenRouter

All clients share a bounded async retry with exponential backoff. They expose a
single `chat(...)` coroutine returning the assistant text, so rollout.py and
judge.py never touch provider-specific wire formats.
"""

from __future__ import annotations

import asyncio
import os

import httpx


class APIError(RuntimeError):
    pass


async def _retrying_post(client: httpx.AsyncClient, url, *, headers, json,
                         max_retries: int, label: str) -> dict:
    """POST with exponential backoff on transient (429/5xx/network) errors."""
    delay = 2.0
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, headers=headers, json=json)
            if resp.status_code in (429, 500, 502, 503, 504, 529):
                raise APIError(f"{label}: HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, APIError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                break
            # jitter via attempt index keeps backoff reproducible-ish without RNG
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise APIError(f"{label}: exhausted {max_retries} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Generation client (models under test)
# ---------------------------------------------------------------------------

class GenerationClient:
    def __init__(self, gen_cfg: dict, client: httpx.AsyncClient):
        self.cfg = gen_cfg
        self.client = client
        self.provider = gen_cfg["provider"]
        self.base_url = gen_cfg["base_url"].rstrip("/")
        if self.provider in ("openrouter", "local_vllm"):
            # local_vllm typically needs no key; openrouter does
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if self.provider == "openrouter" and not self.api_key:
                raise APIError("OPENROUTER_API_KEY is not set")
        else:
            raise APIError(f"unknown generation provider: {self.provider!r}")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter":
            # OpenRouter etiquette headers (optional but recommended)
            h["HTTP-Referer"] = "https://github.com/local/gemma-distress-replication"
            h["X-Title"] = "gemma-distress-replication"
        return h

    async def chat(self, model_id: str, messages: list[dict]) -> str:
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": self.cfg["temperature"],
            "max_tokens": self.cfg["max_tokens"],
        }
        if self.cfg.get("disable_reasoning"):
            # OpenRouter-normalised way to turn off thinking/reasoning tokens.
            # NOTE: Gemini 2.5 Pro may still emit hidden reasoning (paper caveat).
            body["reasoning"] = {"enabled": False}
        data = await _retrying_post(
            self.client, f"{self.base_url}/chat/completions",
            headers=self._headers(), json=body,
            max_retries=self.cfg["max_retries"], label=f"gen[{model_id}]",
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise APIError(f"gen[{model_id}]: unexpected response shape: {data}") from e


# ---------------------------------------------------------------------------
# Judge client (Claude Sonnet 4)
# ---------------------------------------------------------------------------

class JudgeClient:
    def __init__(self, judge_cfg: dict, client: httpx.AsyncClient):
        self.cfg = judge_cfg
        self.client = client
        self.provider = judge_cfg["provider"]
        if self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not self.api_key:
                raise APIError("ANTHROPIC_API_KEY is not set (judge.provider: anthropic)")
            self.base_url = judge_cfg["base_url"].rstrip("/")
            self.model = judge_cfg["model"]
        elif self.provider == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not self.api_key:
                raise APIError("OPENROUTER_API_KEY is not set (judge.provider: openrouter)")
            self.base_url = judge_cfg["openrouter_base_url"].rstrip("/")
            self.model = judge_cfg["openrouter_model"]
        else:
            raise APIError(f"unknown judge provider: {self.provider!r}")

    async def chat(self, prompt: str) -> str:
        if self.provider == "anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": self.cfg["max_tokens"],
                "temperature": self.cfg["temperature"],
                "messages": [{"role": "user", "content": prompt}],
            }
            data = await _retrying_post(
                self.client, f"{self.base_url}/v1/messages",
                headers=headers, json=body,
                max_retries=self.cfg["max_retries"], label="judge",
            )
            try:
                return "".join(
                    block.get("text", "") for block in data["content"]
                    if block.get("type") == "text"
                )
            except (KeyError, TypeError) as e:
                raise APIError(f"judge: unexpected response shape: {data}") from e
        else:  # openrouter (OpenAI-compatible)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": self.cfg["max_tokens"],
                "temperature": self.cfg["temperature"],
                "messages": [{"role": "user", "content": prompt}],
            }
            data = await _retrying_post(
                self.client, f"{self.base_url}/chat/completions",
                headers=headers, json=body,
                max_retries=self.cfg["max_retries"], label="judge",
            )
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise APIError(f"judge: unexpected response shape: {data}") from e
