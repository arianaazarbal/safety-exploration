"""Thin LLM clients for the auxiliary models (judge, onset labeller, paraphraser,
Petri auditor/judge).

These are *not* the evaluation targets — they are the fixed graders the paper
pins. Claude models go through the official Anthropic SDK; OpenAI/Gemini graders
(e.g. the GPT-5-mini judge-agreement check) go through OpenRouter's
OpenAI-compatible endpoint.

Both helpers retry transient failures with exponential backoff and return plain
text. ``extract_json`` pulls the trailing JSON object out of a response that may
contain leading prose (the onset prompt explicitly allows "analysis first, then
JSON").
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .config import (
    ANTHROPIC_API_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
)


def anthropic_complete(
    model: str,
    *,
    system: Optional[str] = None,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """Single completion from a Claude model via the Anthropic SDK.

    ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
    Returns the concatenated text blocks of the response.
    """
    import anthropic

    key = os.environ.get(ANTHROPIC_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {ANTHROPIC_API_KEY_ENV} to call Claude graders")
    client = anthropic.Anthropic(api_key=key)

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as e:  # SDK raises typed errors; backoff and retry
            last_err = e
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Anthropic call failed after {max_retries} retries: {last_err}")


def openrouter_complete(
    model: str,
    *,
    system: Optional[str] = None,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """Single completion via OpenRouter's OpenAI-compatible chat endpoint.

    Used for the GPT-5-mini judge-agreement validation and any non-Claude
    grader. ``system`` is prepended as a system message when provided.
    """
    from openai import OpenAI

    key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {OPENROUTER_API_KEY_ENV} to call OpenRouter graders")
    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)

    full_messages = []
    if system is not None:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=full_messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenRouter call failed after {max_retries} retries: {last_err}")


def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of the last top-level JSON object in ``text``.

    Judge/onset prompts return a JSON object, sometimes preceded by reasoning
    prose. We scan for balanced ``{...}`` spans and parse the last one that
    decodes. Returns ``None`` if no valid object is found (caller decides how to
    handle a malformed grade).
    """
    spans = _balanced_spans(text)
    for start, end in reversed(spans):
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Tolerate the curly-quote variants the paper's prompt examples use.
            cleaned = candidate.replace("“", '"').replace("”", '"')
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None


def _balanced_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) indices of every balanced ``{...}`` span."""
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append((start, i + 1))
    return spans
