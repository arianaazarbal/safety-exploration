"""Lightweight one-shot LLM calls for the auditor persona and the analysis pass.

Separate from providers.py (which manages stateful tool-calling conversations);
this is just "system + user -> text", optionally parsed as JSON.
"""
from __future__ import annotations

import json
from typing import Any


def complete(provider: str, model: str, system: str, user: str,
             max_tokens: int = 1024, temperature: float = 0.7) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    if provider == "openai":
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    raise ValueError(f"unknown provider: {provider}")


def complete_json(provider: str, model: str, system: str, user: str,
                  max_tokens: int = 1024) -> dict[str, Any]:
    """Same as complete() but instructs/parses JSON. Tolerant of code fences."""
    raw = complete(provider, model,
                   system + "\n\nRespond with ONLY a single JSON object.",
                   user, max_tokens=max_tokens, temperature=0.0)
    return _extract_json(raw)


def _extract_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"_parse_error": True, "_raw": raw}
