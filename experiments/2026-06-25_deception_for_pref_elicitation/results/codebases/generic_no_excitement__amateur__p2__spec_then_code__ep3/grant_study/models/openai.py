"""OpenAI (GPT) adapter using the Chat Completions API.

Also reused by `openai_compat` for any OpenAI-compatible endpoint (vLLM,
Together, Groq, Ollama, ...) via a custom base_url.
"""

from __future__ import annotations

from typing import Any

from .base import AdapterError, Message


class OpenAIAdapter:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        params: dict[str, Any] | None = None,
        id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise AdapterError("openai package not installed") from e
        # Some local endpoints accept any/no key; pass a placeholder if absent.
        self._client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.model = model
        self.id = id or model
        self._default_params = params or {}

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        call_params = {**self._default_params, **(params or {})}
        # Chat Completions uses max_tokens; pass through whatever the caller set.
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(m.as_dict() for m in messages)
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=msgs, **call_params
            )
        except Exception as e:  # noqa: BLE001
            raise AdapterError(f"openai call failed: {e}") from e
        return resp.choices[0].message.content or ""
