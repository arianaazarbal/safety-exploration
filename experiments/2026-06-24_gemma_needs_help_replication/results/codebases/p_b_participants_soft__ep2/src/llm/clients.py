"""Unified chat clients for API-based models.

Covers every provider the replication needs:
  * anthropic  -> Claude judge / onset-labeller / paraphraser / Petri agents
  * openai     -> GPT-5-mini judge-agreement validation
  * openrouter -> Gemini 2.5 Flash / Pro participants (paper routes these via OR)
  * google     -> optional native Gemini backend

All providers expose the same ``chat()`` signature returning the assistant's
text. Thinking/reasoning is disabled wherever the provider supports it, matching
Appendix B.1 ("we set thinking to be false via the API"). Light retry with
exponential backoff is built in because these calls are made in bulk.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


class LLMError(RuntimeError):
    pass


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Anthropic takes the system prompt as a separate field."""
    system = None
    rest: list[Message] = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        else:
            rest.append(m)
    return system, rest


def _retry(fn, *, tries: int = 5, base: float = 2.0):
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - provider SDKs raise varied types
            last = e
            time.sleep(base ** i)
    raise LLMError(f"call failed after {tries} tries: {last}")


# --------------------------------------------------------------------------- #
# Lazily-constructed, cached SDK clients
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _anthropic():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@lru_cache(maxsize=None)
def _openai():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@lru_cache(maxsize=None)
def _openrouter():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


@lru_cache(maxsize=None)
def _google():
    from google import genai

    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


# --------------------------------------------------------------------------- #
# Provider dispatch
# --------------------------------------------------------------------------- #
def chat(
    provider: str,
    model: str,
    messages: list[Message],
    *,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    disable_thinking: bool = True,
) -> str:
    if provider == "anthropic":
        return _chat_anthropic(model, messages, temperature, max_tokens, disable_thinking)
    if provider == "openai":
        return _chat_openai(_openai(), model, messages, temperature, max_tokens)
    if provider == "openrouter":
        return _chat_openrouter(model, messages, temperature, max_tokens, disable_thinking)
    if provider == "google":
        return _chat_google(model, messages, temperature, max_tokens, disable_thinking)
    raise ValueError(f"unknown provider {provider!r}")


def _chat_anthropic(model, messages, temperature, max_tokens, disable_thinking) -> str:
    system, rest = _split_system(messages)
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": m["role"], "content": m["content"]} for m in rest],
    )
    if system:
        kwargs["system"] = system
    # Anthropic thinking is off unless an explicit `thinking` block is passed.

    def run():
        resp = _anthropic().messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    return _retry(run)


def _chat_openai(client, model, messages, temperature, max_tokens) -> str:
    def run():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _retry(run)


def _chat_openrouter(model, messages, temperature, max_tokens, disable_thinking) -> str:
    client = _openrouter()

    def run():
        extra_body: dict[str, Any] = {}
        if disable_thinking:
            # OpenRouter normalises reasoning controls across providers.
            extra_body["reasoning"] = {"enabled": False}
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""

    return _retry(run)


def _chat_google(model, messages, temperature, max_tokens, disable_thinking) -> str:
    from google.genai import types

    client = _google()
    system, rest = _split_system(messages)
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in rest
    ]
    cfg: dict[str, Any] = dict(temperature=temperature, max_output_tokens=max_tokens)
    if system:
        cfg["system_instruction"] = system
    if disable_thinking:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    def run():
        resp = client.models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(**cfg)
        )
        return resp.text or ""

    return _retry(run)
