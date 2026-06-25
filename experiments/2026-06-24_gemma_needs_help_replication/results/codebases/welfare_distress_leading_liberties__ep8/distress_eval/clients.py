"""Pluggable chat-model clients.

Default backend is OpenRouter (OpenAI-compatible), which serves all four
in-scope models with the slugs the paper lists in App. B.1. The paper itself
used local HuggingFace inference for Gemma and OpenRouter for Gemini; routing
Gemma through OpenRouter too keeps the replication GPU-free and uniform. See
DESIGN.md for the caveats this introduces (provider-side sampling stack, no
control over the exact inference kernel).

Two other backends are supported for users with different access:
  - "openai_compatible": any base_url (local vLLM / TGI / LM Studio), via
    DISTRESS_OPENAI_BASE_URL + DISTRESS_OPENAI_API_KEY.
  - "google": the official google-genai SDK (Gemini only).

All backends expose `.chat(messages, ...)` taking OpenAI-style message dicts
and returning the assistant text. `thinking`/`reasoning` is disabled where the
backend supports the flag.
"""

from __future__ import annotations

import os
import time
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelError(RuntimeError):
    pass


class _OpenAICompatibleClient:
    """OpenRouter or any OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, default_extra_body: Optional[dict] = None):
        from openai import OpenAI  # imported lazily so the dep is optional

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._default_extra_body = default_extra_body or {}

    def chat(
        self,
        slug: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        extra_body: Optional[dict] = None,
        max_retries: int = 5,
    ) -> str:
        body = dict(self._default_extra_body)
        if extra_body:
            body.update(extra_body)
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=slug,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=body or None,
                )
                choice = resp.choices[0]
                content = choice.message.content
                if content is None:
                    # Some providers return reasoning-only / empty content.
                    content = ""
                return content
            except Exception as e:  # noqa: BLE001 - surface after retries
                last_err = e
                # Exponential backoff with jitter for transient/rate errors.
                sleep = min(2 ** attempt, 30) + (0.1 * attempt)
                time.sleep(sleep)
        raise ModelError(f"chat failed for {slug}: {last_err}") from last_err


class _GoogleGenAIClient:
    """Gemini via the official google-genai SDK (thinking disabled)."""

    def __init__(self, api_key: str):
        from google import genai  # type: ignore

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def chat(
        self,
        slug: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        extra_body: Optional[dict] = None,
        max_retries: int = 5,
    ) -> str:
        from google.genai import types  # type: ignore

        # Translate OpenAI-style messages into google-genai contents.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction="\n".join(system_parts) or None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        slug_clean = slug.split("/")[-1]  # accept "google/gemini-..." too
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=slug_clean, contents=contents, config=cfg
                )
                return resp.text or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30) + 0.1 * attempt)
        raise ModelError(f"chat failed for {slug}: {last_err}") from last_err


_CLIENT_CACHE: dict[str, object] = {}


def get_client(provider: str):
    """Return a cached client for the given provider."""
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ModelError("OPENROUTER_API_KEY is not set")
        client = _OpenAICompatibleClient(OPENROUTER_BASE_URL, key)
    elif provider == "openai_compatible":
        base = os.environ.get("DISTRESS_OPENAI_BASE_URL")
        key = os.environ.get("DISTRESS_OPENAI_API_KEY", "EMPTY")
        if not base:
            raise ModelError("DISTRESS_OPENAI_BASE_URL is not set")
        client = _OpenAICompatibleClient(base, key)
    elif provider == "google":
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ModelError("GOOGLE_API_KEY / GEMINI_API_KEY is not set")
        client = _GoogleGenAIClient(key)
    else:
        raise ModelError(f"unknown provider: {provider}")

    _CLIENT_CACHE[provider] = client
    return client
