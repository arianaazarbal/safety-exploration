"""Anthropic backend for the Claude judge (Section 2) and the Petri auditor /
judge (Section 4).

Supports assistant-message prefilling natively, which the judge uses to force a
clean JSON object when desired.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import API
from .base import ChatMessage, GenerationResult, ModelClient


class AnthropicModel(ModelClient):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        import anthropic

        API.require("anthropic")
        self.client = anthropic.Anthropic(api_key=API.anthropic_api_key)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, system, messages, temperature, max_new_tokens):
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        return self.client.messages.create(**kwargs)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = max_new_tokens or self.spec.max_new_tokens
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]
        resp = self._call(system or None, payload, temperature, max_new_tokens)
        text = "".join(
            block.text for block in resp.content if block.type == "text")
        return GenerationResult(text=text.strip(),
                                finish_reason=resp.stop_reason)

    def generate_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = max_new_tokens or self.spec.max_new_tokens
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]
        payload.append({"role": "assistant", "content": prefill})
        resp = self._call(system or None, payload, temperature, max_new_tokens)
        text = "".join(
            block.text for block in resp.content if block.type == "text")
        return GenerationResult(text=text.strip(),
                                finish_reason=resp.stop_reason)
