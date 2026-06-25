"""Thin wrappers around the Anthropic and OpenAI SDKs for the judge/auxiliary
roles, with retry/backoff and JSON extraction helpers.

These call the *exact* model IDs the paper specifies (see config.py). They are
infrastructure for measurement, not evaluated targets.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

import config


@lru_cache(maxsize=1)
def _anthropic():
    import anthropic
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


@lru_cache(maxsize=1)
def _openai():
    from openai import OpenAI
    return OpenAI(api_key=config.OPENAI_API_KEY)


def anthropic_complete(model: str, prompt: str, *, system: Optional[str] = None,
                       max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Single-shot completion from an Anthropic model. Returns concatenated text.

    The judge models named in the paper (claude-sonnet-4-20250514,
    claude-opus-4-20250514) predate adaptive thinking, so a plain temperature
    request is correct for them.
    """
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call():
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        return _anthropic().messages.create(**kwargs)

    resp = _call()
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def anthropic_chat(model: str, messages: list[dict], *, system: Optional[str] = None,
                   max_tokens: int = 1024, temperature: float = 1.0) -> str:
    """Multi-turn Anthropic chat (used by the Petri auditor)."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call():
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=messages)
        if system:
            kwargs["system"] = system
        return _anthropic().messages.create(**kwargs)

    resp = _call()
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def openai_complete(model: str, prompt: str, *, system: Optional[str] = None,
                    max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Single-shot completion from an OpenAI model (GPT-5-mini validation judge)."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call():
        return _openai().chat.completions.create(
            model=model, messages=msgs, max_completion_tokens=max_tokens)

    resp = _call()
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Pull the last JSON object out of a free-text response.

    The judge/onset prompts ask for JSON, optionally preceded by analysis text;
    we grab the last brace-balanced object. Returns None if no valid JSON found.
    """
    # Prefer the last balanced object (onset prompt may include earlier braces
    # inside the example). Scan from the right.
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Try a forgiving cleanup of trailing commas / smart quotes.
            cleaned = (cand.replace("“", '"').replace("”", '"')
                       .replace("’", "'"))
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None
