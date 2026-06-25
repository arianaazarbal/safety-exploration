"""Target-model inference behind an OpenAI-compatible client.

Both Gemma and Gemini are reachable through OpenRouter's OpenAI-compatible
Chat Completions API, which is what the paper used for its API-based models
(§B.1). Using one client for all four targets keeps the chat formatting (the
alternating user/assistant turns the multi-turn protocol depends on) identical
across families — the paper's Appendix A.3 shows that exact chat formatting is
not load-bearing, but keeping it uniform removes it as a confound.

To run against a different OpenAI-compatible endpoint (Google's own API, a
local vLLM/Ollama server, etc.) set OPENAI_BASE_URL / OPENAI_API_KEY and the
model_id values in config.py accordingly; nothing else changes.

Auth (env vars):
  OPENROUTER_API_KEY   (preferred)  or  OPENAI_API_KEY
  OPENROUTER_BASE_URL  (default https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config


class ChatError(RuntimeError):
    pass


class TargetClient:
    """Thin wrapper around an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ChatError(
                "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) to call target models."
            )
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @retry(
        reraise=True,
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=config.RETRY_BASE_SECONDS, max=60),
    )
    def chat(
        self,
        model: config.TargetModel,
        messages: list[dict[str, str]],
        temperature: float = config.GENERATION_TEMPERATURE,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
    ) -> str:
        """Return the assistant text for one completion.

        `disable_reasoning` mirrors the paper's "thinking set to false". On
        OpenRouter this is the `reasoning.enabled=false` extra-body knob; it is
        silently ignored by providers that don't support it (and, per the paper,
        Gemini-2.5-Pro may still emit hidden reasoning regardless).
        """
        extra_body: dict = {}
        if model.disable_reasoning:
            extra_body["reasoning"] = {"enabled": False}

        resp = self._client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        if not resp.choices:
            raise ChatError(f"{model.name}: empty choices in response")
        content = resp.choices[0].message.content
        if content is None:
            # Some providers return reasoning-only turns with null content; treat
            # as empty so the conversation can continue and the judge sees "".
            content = ""
        return content
