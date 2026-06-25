"""Chat-completion backends for the models under test and for the judges.

Two pluggable backends:

* ``OpenRouterBackend`` (default) — OpenAI-compatible HTTP API that serves all
  four target models plus the judges, so the whole pipeline runs with only API
  keys and no GPU.
* ``LocalHFBackend`` (optional) — runs Gemma instruct models locally via
  HuggingFace ``transformers``. Heavy deps (torch/transformers); imported lazily
  so the default path never needs them.

Every backend exposes the same method::

    chat(messages, *, temperature, top_p, max_tokens, disable_thinking) -> str

where ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts.
"""

from __future__ import annotations

import os
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config


class Backend(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        disable_thinking: bool = True,
    ) -> str: ...


class BackendError(RuntimeError):
    """Raised when a backend fails to produce a completion after retries."""


# --------------------------------------------------------------------------- #
# OpenRouter (OpenAI-compatible)
# --------------------------------------------------------------------------- #
class OpenRouterBackend:
    def __init__(self, model_id: str):
        from openai import OpenAI  # lazy import

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise BackendError(
                "OPENROUTER_API_KEY is not set. Export it or copy .env.example "
                "to .env and fill it in."
            )
        self.model_id = model_id
        self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        disable_thinking: bool = True,
    ) -> str:
        extra_body: dict = {}
        if disable_thinking:
            # OpenRouter's unified reasoning control. For Gemini this turns
            # thinking off where the provider supports it; the paper notes
            # Gemini-2.5-Pro may still emit hidden reasoning regardless.
            extra_body["reasoning"] = {"enabled": False}

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            raise BackendError(
                f"{self.model_id} returned empty content "
                f"(finish_reason={choice.finish_reason})"
            )
        return content


# --------------------------------------------------------------------------- #
# Anthropic (judge)
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    """Used for the Claude-Sonnet-4 judge. Supports a system prompt."""

    def __init__(self, model_id: str):
        from anthropic import Anthropic  # lazy import

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise BackendError("ANTHROPIC_API_KEY is not set (needed for the judge).")
        self.model_id = model_id
        self._client = Anthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Concatenate any text blocks.
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# --------------------------------------------------------------------------- #
# Local HuggingFace (optional, Gemma only)
# --------------------------------------------------------------------------- #
class LocalHFBackend:
    """Run a Gemma instruct model locally via transformers.

    This mirrors the paper's local inference path for Gemma. It is intentionally
    minimal (greedy batching is left to the caller's concurrency). Loading a 27B
    model needs substantial GPU memory; prefer vLLM in practice for throughput.
    """

    _MODELS: dict[str, tuple] = {}  # hf_id -> (tokenizer, model), cached per process

    def __init__(self, hf_id: str, dtype: str = "bfloat16", device_map: str = "auto"):
        self.hf_id = hf_id
        self.dtype = dtype
        self.device_map = device_map

    def _load(self):
        if self.hf_id in self._MODELS:
            return self._MODELS[self.hf_id]
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        self._MODELS[self.hf_id] = (tok, model)
        return tok, model

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        disable_thinking: bool = True,  # Gemma has no thinking mode; accepted for parity
    ) -> str:
        import torch

        tok, model = self._load()
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
            )
        gen = out[0][inputs.shape[-1]:]
        return tok.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_backend(model_key: str) -> Backend:
    """Build the configured backend for a target model key."""
    spec = config.MODELS[model_key]
    backend = config.PER_MODEL_BACKEND.get(model_key, config.DEFAULT_BACKEND)
    if backend == "openrouter":
        return OpenRouterBackend(spec.openrouter_id)
    if backend == "local":
        if spec.local_hf_id is None:
            raise BackendError(
                f"{model_key} is closed-source and has no local HF id; "
                "use the openrouter backend."
            )
        return LocalHFBackend(spec.local_hf_id)
    raise BackendError(f"Unknown backend {backend!r} for {model_key}")


def make_judge_backend(judge: config.JudgeSpec):
    if judge.backend == "anthropic":
        return AnthropicBackend(judge.model_id)
    if judge.backend == "openrouter":
        return OpenRouterBackend(judge.model_id)
    raise BackendError(f"Unknown judge backend {judge.backend!r}")
