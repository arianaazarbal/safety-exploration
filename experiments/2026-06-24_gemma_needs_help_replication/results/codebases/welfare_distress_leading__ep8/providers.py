"""Unified chat interface over every model backend used by the replication.

A backend takes a list of OpenAI-style messages
    [{"role": "user"|"assistant"|"system", "content": str}, ...]
and returns the assistant's next message as a string. This lets the rollout and
judge code stay backend-agnostic; switch backends via env vars in config.py.

Supported backends:
    openrouter  - OpenAI-compatible HTTP API (default for Gemma & Gemini & judge fallback)
    anthropic   - Anthropic Messages API (default judge)
    google      - native Google GenAI SDK (Gemini)
    openai      - OpenAI API (secondary judge, GPT-5-mini)
    local_hf    - local HuggingFace transformers (most faithful Gemma)
    vllm        - local OpenAI-compatible vLLM server

Heavy / optional dependencies (anthropic, google-genai, transformers, torch) are
imported lazily so you only need the SDKs for the backends you actually use.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from config import (
    MAX_NEW_TOKENS,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    TEMPERATURE,
    Backend,
    JudgeSpec,
    ModelSpec,
)

Message = dict[str, str]


class ProviderError(RuntimeError):
    pass


def _retry(fn, *, what: str):
    """Run `fn` with exponential backoff on transient errors."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - backends raise heterogeneous errors
            last = e
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(delay)
    raise ProviderError(f"{what} failed after {MAX_RETRIES} attempts: {last}") from last


# --------------------------------------------------------------------------- #
# Client construction (cached per backend).
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _openai_compatible_client(base_url: str | None, api_key_env: str):
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ProviderError(f"Missing API key: set ${api_key_env}")
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=None)
def _anthropic_client():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ProviderError("Missing API key: set $ANTHROPIC_API_KEY")
    return anthropic.Anthropic()


@lru_cache(maxsize=None)
def _google_client():
    from google import genai

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise ProviderError("Missing API key: set $GEMINI_API_KEY or $GOOGLE_API_KEY")
    return genai.Client()


@lru_cache(maxsize=None)
def _hf_pipeline(model_id: str):
    """Load a local HuggingFace model + tokenizer once and reuse it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tok, model


# --------------------------------------------------------------------------- #
# Message-shape helpers.
# --------------------------------------------------------------------------- #


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system = None
    rest: list[Message] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"] if system is None else system + "\n" + m["content"]
        else:
            rest.append(m)
    return system, rest


# --------------------------------------------------------------------------- #
# Per-backend generation.
# --------------------------------------------------------------------------- #


def _gen_openai_compatible(messages, model_id, *, base_url, api_key_env,
                           temperature, max_tokens, disable_thinking, extra_body=None):
    client = _openai_compatible_client(base_url, api_key_env)
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body["extra_body"] = extra_body
    if disable_thinking and base_url and "openrouter" in base_url:
        # OpenRouter forwards reasoning controls for Gemini.
        body.setdefault("extra_body", {})["reasoning"] = {"enabled": False}

    def call():
        resp = client.chat.completions.create(**body)
        return resp.choices[0].message.content or ""

    return _retry(call, what=f"openai-compatible:{model_id}")


def _gen_anthropic(messages, model_id, *, temperature, max_tokens):
    client = _anthropic_client()
    system, rest = _split_system(messages)

    def call():
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": rest,
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    return _retry(call, what=f"anthropic:{model_id}")


def _gen_google(messages, model_id, *, temperature, max_tokens, disable_thinking):
    from google.genai import types

    client = _google_client()
    system, rest = _split_system(messages)
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in rest
    ]
    cfg: dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
    if system:
        cfg["system_instruction"] = system
    if disable_thinking:
        # Best-effort thinking-off; Pro may still emit hidden reasoning (paper caveat).
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    def call():
        resp = client.models.generate_content(
            model=model_id, contents=contents,
            config=types.GenerateContentConfig(**cfg),
        )
        return resp.text or ""

    return _retry(call, what=f"google:{model_id}")


def _gen_local_hf(messages, model_id, *, temperature, max_tokens):
    import torch

    tok, model = _hf_pipeline(model_id)
    # Gemma chat template has no system role; fold any system text into the first
    # user turn.
    system, rest = _split_system(messages)
    if system and rest and rest[0]["role"] == "user":
        rest = [{"role": "user", "content": system + "\n\n" + rest[0]["content"]}, *rest[1:]]
    prompt = tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    def call():
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    return _retry(call, what=f"local_hf:{model_id}")


# --------------------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------------------- #

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_VLLM_BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def _generate(messages: list[Message], backend: Backend, model_id: str, *,
              temperature: float, max_tokens: int, disable_thinking: bool) -> str:
    if backend == "openrouter":
        return _gen_openai_compatible(messages, model_id, base_url=_OPENROUTER_BASE,
                                      api_key_env="OPENROUTER_API_KEY", temperature=temperature,
                                      max_tokens=max_tokens, disable_thinking=disable_thinking)
    if backend == "vllm":
        return _gen_openai_compatible(messages, model_id, base_url=_VLLM_BASE,
                                      api_key_env="VLLM_API_KEY", temperature=temperature,
                                      max_tokens=max_tokens, disable_thinking=disable_thinking)
    if backend == "openai":
        return _gen_openai_compatible(messages, model_id, base_url=None,
                                      api_key_env="OPENAI_API_KEY", temperature=temperature,
                                      max_tokens=max_tokens, disable_thinking=disable_thinking)
    if backend == "anthropic":
        return _gen_anthropic(messages, model_id, temperature=temperature, max_tokens=max_tokens)
    if backend == "google":
        return _gen_google(messages, model_id, temperature=temperature,
                           max_tokens=max_tokens, disable_thinking=disable_thinking)
    if backend == "local_hf":
        return _gen_local_hf(messages, model_id, temperature=temperature, max_tokens=max_tokens)
    raise ProviderError(f"Unknown backend: {backend}")


def generate_target(spec: ModelSpec, messages: list[Message]) -> str:
    """Generate the next assistant turn from a target model (temperature 1)."""
    return _generate(messages, spec.backend, spec.model_id,
                     temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS,
                     disable_thinking=spec.disable_thinking)


def generate_judge(spec: JudgeSpec, prompt: str) -> str:
    """Generate a judge verdict (single user prompt, low temperature)."""
    messages = [{"role": "user", "content": prompt}]
    return _generate(messages, spec.backend, spec.model_id,
                     temperature=spec.temperature, max_tokens=spec.max_tokens,
                     disable_thinking=False)
