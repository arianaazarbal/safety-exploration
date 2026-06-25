"""Thin helpers for one-shot Claude / OpenRouter completions, used by the
prefill (onset, paraphrase) and Petri modules. Kept separate from the judge
classes so utility callers don't carry judge-specific parsing.
"""

from __future__ import annotations

import os
import time


def anthropic_complete(model: str, prompt: str, *, max_tokens: int = 1024,
                       temperature: float = 0.0, max_retries: int = 5,
                       system: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                          messages=[{"role": "user", "content": prompt}])
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"anthropic_complete failed: {last_err}")


def openrouter_complete(model: str, messages: list[dict], *, max_tokens: int = 1024,
                        temperature: float = 1.0, max_retries: int = 5) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1")
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"openrouter_complete failed: {last_err}")
