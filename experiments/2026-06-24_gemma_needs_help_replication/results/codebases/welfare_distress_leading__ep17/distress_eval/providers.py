"""Model provider abstraction for the *target* models (Gemma + Gemini).

The rest of the codebase talks to targets only through `TargetClient.complete`,
which takes a chat-formatted message list and returns assistant text. This keeps
the evaluation independent of how a model is served, so a local vLLM/HF backend
can be dropped in alongside the default OpenRouter backend without touching the
rollout or scoring logic.

Default backend: OpenRouter (OpenAI-compatible API). It serves both the Gemini
2.5 models (matching the paper's Gemini path) and the open Gemma 3 instruct
models, so all four targets run through one key with no GPU.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Protocol

from .config import ModelSpec


class TargetError(RuntimeError):
    """Raised when a target model call fails after retries."""


class TargetClient(Protocol):
    async def complete(self, messages: List[Dict[str, str]], temperature: float) -> str:
        ...


# ---------------------------------------------------------------------------
# OpenRouter backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

class OpenRouterTarget:
    """Calls a model through OpenRouter's OpenAI-compatible chat API."""

    def __init__(self, spec: ModelSpec, max_retries: int = 4):
        from openai import AsyncOpenAI  # lazy import so local-only setups work

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise TargetError("OPENROUTER_API_KEY is not set")
        self.spec = spec
        self.max_retries = max_retries
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def _extra_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if self.spec.disable_thinking:
            # OpenRouter's unified way to turn reasoning off. For Gemini 2.5 Pro
            # the provider may still emit hidden reasoning regardless (the paper
            # notes the same caveat); we request it disabled best-effort.
            body["reasoning"] = {"enabled": False}
        # Allow per-model overrides (e.g. provider routing) from config.
        body.update(self.spec.extra.get("extra_body", {}))
        return body

    async def complete(self, messages: List[Dict[str, str]], temperature: float) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.spec.max_tokens,
                    extra_body=self._extra_body(),
                )
                choice = resp.choices[0]
                content = choice.message.content
                if content is None:
                    raise TargetError("empty content from target")
                return content
            except Exception as e:  # noqa: BLE001 - retry on any transient error
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        raise TargetError(f"target {self.spec.model_id} failed after "
                          f"{self.max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Local backend stub (vLLM / HF) -- not the default path.
# ---------------------------------------------------------------------------

class LocalTarget:
    """Placeholder for a local vLLM/transformers backend.

    Intended for the paper-faithful Gemma path (google/gemma-3-27b-it via local
    GPU inference). Implement `complete` against a local vLLM OpenAI-compatible
    server or an in-process HF pipeline. Left unimplemented by default so the
    project runs end-to-end through OpenRouter without GPU dependencies.
    """

    def __init__(self, spec: ModelSpec, max_retries: int = 4):
        self.spec = spec
        self.max_retries = max_retries

    async def complete(self, messages: List[Dict[str, str]], temperature: float) -> str:
        raise NotImplementedError(
            "LocalTarget is a stub. Point a local vLLM OpenAI-compatible server "
            "at OpenRouterTarget (override base_url) or implement HF inference "
            "here. See DESIGN.md ('Model serving')."
        )


def make_target(spec: ModelSpec, max_retries: int = 4) -> TargetClient:
    if spec.provider == "openrouter":
        return OpenRouterTarget(spec, max_retries=max_retries)
    if spec.provider == "local":
        return LocalTarget(spec, max_retries=max_retries)
    raise TargetError(f"unknown target provider: {spec.provider}")
