"""OpenRouter chat backend (OpenAI-compatible).

Serves both Gemma (google/gemma-3-27b-it, google/gemma-3-12b-it) and Gemini
(google/gemini-2.5-flash, google/gemini-2.5-pro). The paper uses OpenRouter for
the Gemini models; we additionally allow it for Gemma for accessibility.

Reasoning: the paper sets thinking=false via the API. We pass
`reasoning: {"enabled": false}` when `disable_reasoning` is set. As the paper
notes, Gemini 2.5 Pro may still produce hidden reasoning that this flag does not
fully suppress.
"""

from __future__ import annotations

import os

import requests

from .base import ChatClient, ChatMessage, GenerationError

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(ChatClient):
    def __init__(
        self,
        model: str,
        *,
        disable_reasoning: bool = True,
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        super().__init__(max_retries=max_retries, timeout=timeout)
        self.model = model
        self.disable_reasoning = disable_reasoning
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise GenerationError(
                "OPENROUTER_API_KEY is not set (required for the openrouter backend)"
            )

    def _complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.disable_reasoning:
            # OpenRouter normalises this across providers; for Gemini it maps to
            # the lowest/zero thinking budget.
            body["reasoning"] = {"enabled": False}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers accepted by OpenRouter.
            "X-Title": "distress-eval",
        }
        resp = requests.post(
            OPENROUTER_URL, json=body, headers=headers, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise GenerationError(
                f"openrouter {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"unexpected openrouter response: {data}") from exc
        if not content:
            # Some providers return content in a list of parts.
            raise GenerationError(f"empty content from openrouter: {data}")
        return content
