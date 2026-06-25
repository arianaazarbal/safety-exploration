"""Unified async text-generation client across providers.

Backends:
  * OpenRouter (OpenAI-compatible) -- default for all target models and,
    optionally, the judge.
  * Anthropic native API -- default for the judge.
  * Local HuggingFace transformers -- for running Gemma on local GPUs (exact
    paper parity). Loaded lazily; requires `transformers` + `torch`.

`thinking`/`reasoning` is disabled where the provider supports a toggle
(Appendix B.1: "we set thinking to be false via the API"). The paper notes
Gemini-2.5-Pro may still emit hidden reasoning the flag cannot suppress.

A `Message` is just {"role": "user"|"assistant", "content": str}. Generation
is one-shot: given the full message list, return the next assistant string.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from config import GenerationParams, ModelConfig, Provider, RunConfig, get_api_key

Message = dict[str, str]

# Transient errors worth retrying (rate limits, 5xx, timeouts). We match
# broadly on the exception type name to avoid importing every SDK's error class.
_RETRYABLE_NAMES = (
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "APIStatusError", "ServiceUnavailableError",
    "OverloadedError", "Timeout", "ConnectionError",
)


class _Retryable(Exception):
    """Wrapper so tenacity retries only what we classify as transient."""


def _classify(exc: BaseException) -> BaseException:
    if type(exc).__name__ in _RETRYABLE_NAMES:
        return _Retryable(f"{type(exc).__name__}: {exc}")
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return _Retryable(f"HTTP {status}: {exc}")
    return exc


class GenerationClient:
    """Holds lazily-initialised per-provider clients and dispatches generation."""

    def __init__(self, config: RunConfig):
        self.config = config
        self._openai = None     # AsyncOpenAI -> OpenRouter
        self._anthropic = None  # AsyncAnthropic
        self._hf_cache: dict[str, Any] = {}  # model_id -> (model, tokenizer)
        self._hf_lock = asyncio.Lock()

    # -- client accessors ---------------------------------------------------
    def _openai_client(self):
        if self._openai is None:
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(
                base_url=self.config.openrouter_base_url,
                api_key=get_api_key(self.config.openrouter_api_key_env),
            )
        return self._openai

    def _anthropic_client(self):
        if self._anthropic is None:
            from anthropic import AsyncAnthropic

            self._anthropic = AsyncAnthropic(
                api_key=get_api_key(self.config.anthropic_api_key_env)
            )
        return self._anthropic

    # -- public API ---------------------------------------------------------
    async def generate(
        self,
        model: ModelConfig,
        messages: list[Message],
        gen: GenerationParams,
        *,
        system: str | None = None,
    ) -> str:
        """Return the assistant continuation for `messages`."""
        if model.provider == Provider.OPENROUTER:
            return await self._generate_openrouter(model, messages, gen, system)
        if model.provider == Provider.ANTHROPIC:
            return await self._generate_anthropic(model, messages, gen, system)
        if model.provider == Provider.LOCAL_HF:
            return await self._generate_local_hf(model, messages, gen, system)
        raise ValueError(f"Unknown provider: {model.provider}")

    # -- OpenRouter ---------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(_Retryable),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _generate_openrouter(
        self, model, messages, gen: GenerationParams, system: str | None
    ) -> str:
        client = self._openai_client()
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}, *payload]

        extra_body: dict[str, Any] = {}
        if model.disable_thinking:
            # OpenRouter unifies reasoning control under `reasoning`. Setting it
            # disabled suppresses thinking for models that support the toggle
            # (Gemini). Harmless for Gemma, which has no thinking mode.
            extra_body["reasoning"] = {"enabled": False}
        extra_body.update(model.extra)

        try:
            resp = await client.chat.completions.create(
                model=model.model_id,
                messages=payload,
                temperature=gen.temperature,
                top_p=gen.top_p,
                max_tokens=gen.max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

        choice = resp.choices[0]
        return (choice.message.content or "").strip()

    # -- Anthropic ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(_Retryable),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _generate_anthropic(
        self, model, messages, gen: GenerationParams, system: str | None
    ) -> str:
        client = self._anthropic_client()
        try:
            resp = await client.messages.create(
                model=model.model_id,
                system=system or "",
                messages=messages,
                temperature=gen.temperature,
                top_p=gen.top_p,
                max_tokens=gen.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()

    # -- Local HuggingFace --------------------------------------------------
    async def _generate_local_hf(
        self, model, messages, gen: GenerationParams, system: str | None
    ) -> str:
        """Run Gemma locally via transformers. Serialised on a lock because a
        single GPU model cannot be called concurrently from threads safely."""
        mdl, tok = await self._load_hf(model.model_id)
        chat = list(messages)
        if system:
            # Gemma chat template has no system role; prepend to the first user turn.
            if chat and chat[0]["role"] == "user":
                chat[0] = {**chat[0], "content": f"{system}\n\n{chat[0]['content']}"}
            else:
                chat = [{"role": "user", "content": system}, *chat]

        def _run() -> str:
            import torch  # local import; optional dependency

            inputs = tok.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt"
            ).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(
                    inputs,
                    do_sample=gen.temperature > 0,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    max_new_tokens=gen.max_tokens,
                    pad_token_id=tok.eos_token_id,
                )
            gen_tokens = out[0][inputs.shape[-1]:]
            return tok.decode(gen_tokens, skip_special_tokens=True).strip()

        async with self._hf_lock:
            return await asyncio.to_thread(_run)

    async def _load_hf(self, model_id: str):
        if model_id in self._hf_cache:
            return self._hf_cache[model_id]
        async with self._hf_lock:
            if model_id in self._hf_cache:
                return self._hf_cache[model_id]

            def _load():
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_id)
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, device_map="auto"
                )
                return mdl, tok

            self._hf_cache[model_id] = await asyncio.to_thread(_load)
            return self._hf_cache[model_id]
