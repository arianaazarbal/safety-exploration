"""API-backed chat clients: OpenAI-compatible (OpenRouter/Gemini, GPT-5-mini) and
Anthropic (Claude judge / Petri auditor & judge).

Both wrap their SDK with retry/backoff and expose the shared :class:`ChatModel`
surface. Thinking/reasoning is disabled where the API allows it (Appendix B.1).
Only Anthropic supports assistant prefill; OpenAI-compatible backends raise on
prefill (Gemini has no base model, so the Section 3 prefill path never needs them).
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import API, GENERATION, GenerationConfig, ModelSpec
from .base import GenResult, Message


def _retry():
    return retry(
        stop=stop_after_attempt(API.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )


class OpenAICompatModel:
    """Gemini (via OpenRouter) and GPT-5-mini (via OpenAI), both through the
    OpenAI Chat Completions schema."""

    def __init__(self, spec: ModelSpec, **kwargs) -> None:
        from openai import OpenAI

        self.spec = spec
        self.spec_key = spec.key

        if spec.backend == "openrouter":
            api_key = os.environ.get(API.openrouter_api_key_env)
            base_url = API.openrouter_base_url
        else:  # "openai"
            api_key = os.environ.get(API.openai_api_key_env)
            base_url = os.environ.get(API.openai_base_url_env) or None

        if not api_key:
            env = (API.openrouter_api_key_env if spec.backend == "openrouter"
                   else API.openai_api_key_env)
            raise RuntimeError(f"Missing API key: set ${env} for model {spec.key!r}")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=API.request_timeout_s)

    def supports_prefill(self) -> bool:
        return False

    def _extra_body(self, gen: GenerationConfig) -> dict:
        """Disable hidden reasoning where the provider honours it (Appendix B.1).

        OpenRouter exposes a unified ``reasoning`` control; for Gemini this maps
        to disabling thinking. Providers that ignore it (Gemini-2.5-Pro,
        GPT-5.2) may still emit hidden reasoning -- a caveat the paper notes too.
        """
        if not gen.disable_thinking:
            return {}
        if self.spec.backend == "openrouter":
            return {"reasoning": {"enabled": False}}
        return {}

    @_retry()
    def generate(
        self,
        messages: list[Message],
        *,
        gen: GenerationConfig = GENERATION,
        prefill: str | None = None,
    ) -> GenResult:
        if prefill is not None:
            raise NotImplementedError(
                f"{self.spec.backend} backend does not support assistant prefill"
            )
        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=[m.to_dict() for m in messages],
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            extra_body=self._extra_body(gen),
        )
        choice = resp.choices[0]
        usage = resp.usage
        return GenResult(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
        )


class AnthropicModel:
    """Claude models (judge, Petri auditor, Petri judge). Supports prefill via a
    trailing assistant message."""

    def __init__(self, spec: ModelSpec, **kwargs) -> None:
        import anthropic

        self.spec = spec
        self.spec_key = spec.key
        api_key = os.environ.get(API.anthropic_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set ${API.anthropic_api_key_env} for {spec.key!r}"
            )
        self.client = anthropic.Anthropic(api_key=api_key, timeout=API.request_timeout_s)

    def supports_prefill(self) -> bool:
        return True

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict]]:
        system = None
        convo: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                convo.append(m.to_dict())
        return system, convo

    @_retry()
    def generate(
        self,
        messages: list[Message],
        *,
        gen: GenerationConfig = GENERATION,
        prefill: str | None = None,
    ) -> GenResult:
        system, convo = self._split_system(messages)
        if prefill is not None:
            convo = convo + [{"role": "assistant", "content": prefill}]

        kwargs: dict = dict(
            model=self.spec.model_id,
            messages=convo,
            max_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
        )
        if system:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return GenResult(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            finish_reason=resp.stop_reason,
        )
