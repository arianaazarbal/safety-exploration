"""API backends: Gemini (target), Claude and OpenAI-compatible (auxiliary).

Generation only — no prefill, no logits, not fine-tunable.  Concurrency is the
caller's responsibility (the rollout/judge layers fan out with a thread pool);
this class is safe to share across threads because the underlying SDK clients
are thread-safe and each call is independent.

Retries with exponential backoff wrap every request (transient 429/5xx).
"""

from __future__ import annotations

import os

from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from ..config import ModelSpec, RuntimeConfig, SamplingConfig
from .base import ChatBackend, GenerationResult, Message


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system = None
    rest: list[Message] = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        else:
            rest.append(m)
    return system, rest


class APIBackend(ChatBackend):
    def __init__(self, spec: ModelSpec, runtime: RuntimeConfig,
                 gemini_provider: str = "openrouter"):
        super().__init__(spec)
        self.runtime = runtime
        self.gemini_provider = gemini_provider
        self._client = None
        self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        if self.spec.backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(max_retries=0)
        elif self.spec.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif self.spec.backend == "gemini":
            if self.gemini_provider == "openrouter":
                from openai import OpenAI
                self._client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                )
            else:  # google native
                from google import genai
                self._client = genai.Client()
        else:  # pragma: no cover
            raise ValueError(f"APIBackend cannot serve backend={self.spec.backend}")

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        sampling: SamplingConfig,
        n: int = 1,
        prefill: str | None = None,
    ) -> list[GenerationResult]:
        if prefill is not None and not self.spec.supports_prefill:
            raise NotImplementedError(
                f"{self.spec.name}: prefill is not available via this API "
                f"backend (closed-weight). Sections 3-4 are Gemma-only."
            )
        return [self._one(messages, sampling) for _ in range(n)]

    # -- per-provider single completion (retried) -----------------------
    def _one(self, messages: list[Message], sampling: SamplingConfig) -> GenerationResult:
        fn = {
            "anthropic": self._anthropic,
            "openai": self._openai_compatible,
            "gemini": (self._openai_compatible
                       if self.gemini_provider == "openrouter"
                       else self._google),
        }[self.spec.backend]
        return self._with_retry(fn, messages, sampling)

    def _with_retry(self, fn, messages, sampling):
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.runtime.api_max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(Exception),
        )
        def _call():
            return fn(messages, sampling)
        return _call()

    # -- Anthropic (Claude judge / auditor / paraphraser) ---------------
    def _anthropic(self, messages, sampling) -> GenerationResult:
        system, rest = _split_system(messages)
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=sampling.max_new_tokens,
            messages=[{"role": m["role"], "content": m["content"]} for m in rest],
        )
        if system:
            kwargs["system"] = system
        # Auxiliary Claude calls (judge/paraphrase) run without extended
        # thinking; current snapshots accept `temperature` only on older
        # models, so we pass it defensively and let the SDK reject if needed.
        try:
            resp = self._client.messages.create(temperature=sampling.temperature, **kwargs)
        except Exception:
            resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return GenerationResult(text=text, finish_reason=resp.stop_reason)

    # -- OpenAI-compatible (GPT-5-mini judge, Gemini-via-OpenRouter) ----
    def _openai_compatible(self, messages, sampling) -> GenerationResult:
        extra = {}
        if self.spec.backend == "gemini" and self.spec.thinking_disabled:
            # Ask OpenRouter to disable Gemini's thinking where supported.
            extra["extra_body"] = {"reasoning": {"enabled": False}}
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_new_tokens,
            **extra,
        )
        choice = resp.choices[0]
        return GenerationResult(text=choice.message.content or "",
                                finish_reason=choice.finish_reason)

    # -- Google GenAI native --------------------------------------------
    def _google(self, messages, sampling) -> GenerationResult:
        from google.genai import types
        system, rest = _split_system(messages)
        contents = []
        for m in rest:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role,
                                           parts=[types.Part(text=m["content"])]))
        cfg = types.GenerateContentConfig(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_output_tokens=sampling.max_new_tokens,
            system_instruction=system,
            # thinking_budget=0 disables Gemini 2.5 thinking where supported.
            thinking_config=types.ThinkingConfig(thinking_budget=0)
            if self.spec.thinking_disabled else None,
        )
        # google-genai strips the OpenRouter-style "google/" prefix.
        model_id = self.spec.model_id.split("/")[-1]
        resp = self._client.models.generate_content(
            model=model_id, contents=contents, config=cfg,
        )
        return GenerationResult(text=resp.text or "", finish_reason=None)
