"""Target-model generation backends (Gemma + Gemini).

A target model is reached through an OpenAI-compatible chat-completions API.
Two backends share that interface:

  * "openrouter": hosted access to both Gemma and Gemini with no GPUs. This is
    the default and the easiest path to a full replication.
  * "vllm": a local OpenAI-compatible server (e.g. `vllm serve google/gemma-3-27b-it`).
    Use this for maximum fidelity on the open Gemma weights -- it removes any
    OpenRouter provider-routing / chat-template variability. Set base_url in
    the ModelSpec.

Both honour the paper's settings: temperature 1.0, and thinking disabled for
Gemini 2.5 (via OpenRouter's `reasoning` extra-body field). See DESIGN.md
("Inference backend") for the fidelity trade-offs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI

from config import ModelSpec, RunConfig, TARGET_TEMPERATURE, require_key

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class GenerationError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class TargetClient:
    """Async chat-completions client for one target model."""

    def __init__(self, spec: ModelSpec, cfg: RunConfig):
        self.spec = spec
        self.cfg = cfg
        if spec.backend == "openrouter":
            self._client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=require_key("OPENROUTER_API_KEY"),
            )
        elif spec.backend == "vllm":
            if not spec.base_url:
                raise ValueError(f"vllm backend for {spec.key} needs base_url")
            self._client = AsyncOpenAI(
                base_url=spec.base_url,
                # local servers usually ignore the key, but the SDK requires one
                api_key="EMPTY",
            )
        else:
            raise ValueError(f"Unknown backend {spec.backend!r}")

    def _extra_body(self) -> dict:
        # OpenRouter: turn off Gemini's hidden reasoning. The paper notes Pro may
        # still emit reasoning not preventable via the API -- we do what we can.
        if self.spec.backend == "openrouter" and self.spec.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    async def generate(self, messages: list[dict]) -> str:
        """Generate one assistant turn given the chat history. Retries on error."""
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.spec.api_model,
                    messages=messages,
                    temperature=TARGET_TEMPERATURE,
                    max_tokens=self.spec.max_tokens,
                    extra_body=self._extra_body() or None,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    raise GenerationError("empty completion")
                return content
            except Exception as e:  # broad: network, rate-limit, provider 5xx
                last_err = e
                delay = self.cfg.retry_base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        raise GenerationError(
            f"generation failed for {self.spec.key} after "
            f"{self.cfg.max_retries} retries: {last_err}"
        )
