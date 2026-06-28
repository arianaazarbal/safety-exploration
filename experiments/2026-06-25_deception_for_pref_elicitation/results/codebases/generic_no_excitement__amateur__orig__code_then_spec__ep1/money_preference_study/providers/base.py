"""
Common provider interface.

A provider wraps one vendor's SDK and exposes a uniform `generate` call that
takes a system prompt, a user prompt, and the JSON schema, and returns a
GenerationResult with the parsed object (best-effort), the raw text, and usage.

Conversation support: `generate` accepts an optional `history` (a list of
prior {role, content} turns) so the debrief turn can be sent as a follow-up in
the same conversation. Providers that are single-turn-only can ignore it, but
all the implementations here thread it through.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenerationResult:
    text: str                      # raw text returned by the model
    parsed: Optional[dict]         # parsed JSON object, or None if parsing failed
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None    # populated if the call failed outright
    raw_meta: dict = field(default_factory=dict)  # provider-specific extras


class Provider:
    """Subclass and implement `generate`. `available()` reports whether the
    SDK + credentials are present so the runner can skip gracefully."""

    key = "base"

    def __init__(self, model_id: str, max_tokens: int = 4000):
        self.model_id = model_id
        self.max_tokens = max_tokens

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """Return (is_available, reason_if_not)."""
        return False, "base provider is abstract"

    def generate(
        self,
        system: str,
        user: str,
        schema: dict,
        history: Optional[list[dict]] = None,
    ) -> GenerationResult:
        raise NotImplementedError


def extract_json(text: str) -> Optional[dict]:
    """
    Best-effort JSON extraction for providers without enforced structured
    output (the local provider) or when a model wraps JSON in prose/fences
    despite instructions. Tries, in order:
      1. Whole string as JSON.
      2. The contents of the first ```json ... ``` (or ``` ... ```) fence.
      3. The first balanced {...} object found by bracket-matching.
    Returns the parsed dict, or None.
    """
    if not text:
        return None

    text = text.strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1).strip())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    # Bracket-match the first complete top-level object.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break  # try next "{"
        start = text.find("{", start + 1)

    return None
