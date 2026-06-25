"""Chat-completion adapters for the three provider styles we use.

All adapters implement `generate(messages, temperature, max_tokens) -> str`,
where `messages` is a list of {"role": "user"|"assistant", "content": str}.
A system string may be supplied separately; adapters that lack a system role
(Gemma) fold it into the first user message.

Adapters are intentionally thin and synchronous; concurrency is handled by the
runner with a thread pool. Retries/backoff wrap the network call.
"""
from __future__ import annotations

from typing import Any

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .config import ModelSpec

Message = dict[str, str]


class BackendError(RuntimeError):
    pass


def _fold_system_into_first_user(messages: list[Message], system: str) -> list[Message]:
    out = list(messages)
    for i, m in enumerate(out):
        if m["role"] == "user":
            out[i] = {"role": "user", "content": f"{system}\n\n{m['content']}"}
            return out
    # No user message (shouldn't happen): prepend one.
    return [{"role": "user", "content": system}, *out]


class ChatClient:
    """Base interface."""

    def __init__(self, spec: ModelSpec, max_retries: int = 6):
        self.spec = spec
        self.max_retries = max_retries

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        system: str | None = None,
    ) -> str:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# OpenAI-compatible (OpenRouter, vLLM, Together, OpenAI, ...)
# -----------------------------------------------------------------------------
class OpenAICompatClient(ChatClient):
    def __init__(self, spec: ModelSpec, max_retries: int = 6):
        super().__init__(spec, max_retries)
        from openai import OpenAI

        self._client = OpenAI(base_url=spec.base_url, api_key=spec.api_key() or "EMPTY")

    def generate(self, messages, *, temperature, max_tokens, top_p=1.0, system=None):
        msgs: list[Message] = list(messages)
        if system:
            if self.spec.supports_system:
                msgs = [{"role": "system", "content": system}, *msgs]
            else:
                msgs = _fold_system_into_first_user(msgs, system)

        extra_body: dict[str, Any] = {}
        if self.spec.disable_reasoning:
            # OpenRouter-style reasoning suppression. Harmless for providers
            # that ignore it. See DESIGN.md ("Disabling thinking").
            extra_body["reasoning"] = {"enabled": False}

        @retry(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.spec.model,
                messages=msgs,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise BackendError(f"{self.spec.model}: empty content in response")
            return content

        return _call()


# -----------------------------------------------------------------------------
# Google GenAI (direct Gemini / Gemma via AI Studio) -- optional path
# -----------------------------------------------------------------------------
class GeminiClient(ChatClient):
    def __init__(self, spec: ModelSpec, max_retries: int = 6):
        super().__init__(spec, max_retries)
        from google import genai  # type: ignore

        self._genai = genai
        self._client = genai.Client(api_key=spec.api_key())

    def generate(self, messages, *, temperature, max_tokens, top_p=1.0, system=None):
        from google.genai import types  # type: ignore

        msgs = list(messages)
        # Gemma served by Google has no system role -> fold in.
        if system and not self.spec.supports_system:
            msgs = _fold_system_into_first_user(msgs, system)

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in msgs
        ]

        cfg_kwargs: dict[str, Any] = dict(
            temperature=temperature, top_p=top_p, max_output_tokens=max_tokens
        )
        if system and self.spec.supports_system:
            cfg_kwargs["system_instruction"] = system
        if self.spec.disable_reasoning:
            # thinking_budget=0 disables Gemini "thinking" where supported.
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        config = types.GenerateContentConfig(**cfg_kwargs)

        @retry(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        def _call() -> str:
            resp = self._client.models.generate_content(
                model=self.spec.model, contents=contents, config=config
            )
            text = getattr(resp, "text", None)
            if not text:
                raise BackendError(f"{self.spec.model}: empty content in response")
            return text

        return _call()


# -----------------------------------------------------------------------------
# Anthropic (judge)
# -----------------------------------------------------------------------------
class AnthropicClient(ChatClient):
    def __init__(self, spec: ModelSpec, max_retries: int = 6):
        super().__init__(spec, max_retries)
        import anthropic

        self._client = anthropic.Anthropic(api_key=spec.api_key())

    def generate(self, messages, *, temperature, max_tokens, top_p=1.0, system=None):
        @retry(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        def _call() -> str:
            kwargs: dict[str, Any] = dict(
                model=self.spec.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        return _call()


_BACKENDS = {
    "openai": OpenAICompatClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
}


def build_client(spec: ModelSpec, max_retries: int = 6) -> ChatClient:
    if spec.backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{spec.backend}' for {spec.name}")
    return _BACKENDS[spec.backend](spec, max_retries=max_retries)
