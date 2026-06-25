"""OpenRouter backend (OpenAI-compatible) for target models: hosted Gemma-3 and
closed-source Gemini-2.5.

Thinking is disabled where the provider supports it (paper: "we set thinking to
be false via the API", with the caveat that Gemini-2.5-Pro may still produce
hidden reasoning). Sampling is always temperature 1 in the protocol.
"""
from __future__ import annotations

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, ModelSpec
from .base import ChatBackend, ChatMessage


class OpenRouterBackend(ChatBackend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        if spec.openrouter_id is None:
            raise ValueError(f"{spec.name} has no openrouter_id")
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; cannot run OpenRouter backend."
            )
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL
        )
        self.model_id = spec.openrouter_id

    def _extra_body(self) -> dict:
        """Provider-specific knobs to suppress reasoning where possible."""
        body: dict = {}
        if self.spec.family == "gemini":
            # Gemini thinking budget: 0 disables visible reasoning.
            body["reasoning"] = {"max_tokens": 0}
        else:
            # Gemma has no reasoning mode; nothing to disable.
            body["reasoning"] = {"exclude": True}
        return body

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[m.as_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        """OpenRouter exposes prefill via a trailing assistant message for some
        providers. Gemini does not reliably support assistant-prefill
        continuation, so the prefill experiments (Section 3) should use the
        local Gemma backend. We implement the trailing-assistant form for
        completeness and strip the echoed prefill if the provider returns it."""
        convo = list(messages) + [ChatMessage("assistant", prefill)]
        out = self.generate(convo, temperature=temperature, max_tokens=max_tokens)
        if out.startswith(prefill):
            out = out[len(prefill):]
        return out
