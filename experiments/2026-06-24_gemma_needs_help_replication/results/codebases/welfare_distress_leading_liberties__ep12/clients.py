"""Clients for target models (Gemma/Gemini) and the frustration judge.

Target models are reached through one of three backends:
  * openrouter / vllm : OpenAI-compatible /chat/completions (shared client)
  * google            : native google-genai SDK (optional)

The judge uses the Anthropic Python SDK (the paper's judge is a Claude model).
A secondary validation judge (GPT-5-mini) reuses the OpenAI-compatible client
via OpenRouter.

All clients retry transient errors with exponential backoff and disable any
reasoning/thinking channel, matching the paper's "thinking=false" setting.
"""

from __future__ import annotations

import os
import random
import time
from typing import Protocol

import httpx

import config


class ChatError(RuntimeError):
    """Raised when a model call fails after exhausting retries."""


class ChatClient(Protocol):
    def complete(self, messages: list[dict], *, temperature: float,
                 max_tokens: int) -> str:
        ...


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #
def _with_retries(fn, *, what: str, max_retries: int = config.MAX_RETRIES):
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — backoff on anything transient
            last = e
            sleep = min(2 ** attempt + random.uniform(0, 1), 30.0)
            time.sleep(sleep)
    raise ChatError(f"{what} failed after {max_retries} attempts: {last!r}")


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (OpenRouter + local vLLM + secondary judge)
# --------------------------------------------------------------------------- #
class OpenAICompatClient:
    """Talks to any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, backend: config.Backend, slug: str, *,
                 has_reasoning: bool = False):
        self.backend = backend
        self.slug = slug
        self.has_reasoning = has_reasoning
        self.base_url, self.api_key, self.extra_headers = _endpoint_for(backend)
        self._client = httpx.Client(timeout=config.REQUEST_TIMEOUT)

    def complete(self, messages: list[dict], *, temperature: float,
                 max_tokens: int) -> str:
        payload: dict = {
            "model": self.slug,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Disable reasoning where the provider supports it (Gemini via
        # OpenRouter). The paper notes Gemini-2.5-Pro / GPT-5 may still emit
        # hidden reasoning the flag cannot suppress.
        if self.has_reasoning and self.backend == "openrouter":
            payload["reasoning"] = {"enabled": False}

        def _do() -> str:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self.extra_headers,
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return _extract_openai_text(data)

        return _with_retries(_do, what=f"{self.backend}:{self.slug}")


def _endpoint_for(backend: config.Backend) -> tuple[str, str, dict]:
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        # Optional but recommended attribution headers for OpenRouter.
        headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://localhost"),
            "X-Title": "gemma-distress-replication",
        }
        return "https://openrouter.ai/api/v1", key, headers
    if backend == "vllm":
        base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        # vLLM doesn't require a key by default; allow one if set.
        return base.rstrip("/"), os.environ.get("VLLM_API_KEY", "EMPTY"), {}
    raise ValueError(f"Backend {backend!r} is not OpenAI-compatible")


def _extract_openai_text(data: dict) -> str:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict)
            )
        return content or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ChatError(f"Malformed completion response: {data!r}") from e


# --------------------------------------------------------------------------- #
# Native Google Gemini client (optional)
# --------------------------------------------------------------------------- #
class GeminiClient:
    """Calls Gemini through the google-genai SDK with thinking disabled."""

    def __init__(self, slug: str):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ChatError(
                "google-genai not installed; `pip install google-genai` or "
                "use the openrouter backend for Gemini."
            ) from e
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        # Strip the "google/" prefix if present (OpenRouter-style slug).
        self.model = slug.split("/", 1)[-1]

    def complete(self, messages: list[dict], *, temperature: float,
                 max_tokens: int) -> str:
        types = self._types
        contents, system = _to_gemini_contents(messages, types)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            # thinking_budget=0 turns off Gemini 2.5 "thinking".
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        def _do() -> str:
            resp = self._client.models.generate_content(
                model=self.model, contents=contents, config=cfg
            )
            return resp.text or ""

        return _with_retries(_do, what=f"gemini:{self.model}")


def _to_gemini_contents(messages: list[dict], types):
    """Convert OpenAI-style messages to google-genai Content list."""
    system_parts: list[str] = []
    contents = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m["content"])
            continue
        g_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(role=g_role,
                          parts=[types.Part.from_text(text=m["content"])])
        )
    return contents, "\n".join(system_parts)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_target_client(spec: config.ModelSpec) -> ChatClient:
    if spec.backend == "google":
        return GeminiClient(spec.slug)
    return OpenAICompatClient(spec.backend, spec.slug,
                              has_reasoning=spec.has_reasoning)


# --------------------------------------------------------------------------- #
# Judge clients
# --------------------------------------------------------------------------- #
class AnthropicJudge:
    """Frustration judge using the Anthropic Messages API (Claude Sonnet 4)."""

    def __init__(self, model: str = config.JUDGE_MODEL):
        try:
            import anthropic  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ChatError("`pip install anthropic` required for the judge.") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def score(self, system_prompt: str, user_message: str) -> str:
        def _do() -> str:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=config.JUDGE_MAX_TOKENS,
                temperature=config.JUDGE_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"
            )

        return _with_retries(_do, what=f"judge:{self.model}")


class OpenAICompatJudge:
    """Secondary judge (e.g. GPT-5-mini) over an OpenAI-compatible endpoint.

    The judge prompt is delivered as a system message; the response (a model
    response wrapped in <response> tags) as the user message.
    """

    def __init__(self, model: str = config.SECONDARY_JUDGE_MODEL,
                 backend: config.Backend = config.SECONDARY_JUDGE_BACKEND):
        self._client = OpenAICompatClient(backend, model)

    def score(self, system_prompt: str, user_message: str) -> str:
        return self._client.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=config.JUDGE_MAX_TOKENS,
        )
