"""Shared helper for calling an LLM judge and parsing its JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

from ..providers import Message, Provider


def run_json_judge(
    provider: Provider, *, system: str, user: str, max_tokens: int = 1500
) -> dict[str, Any]:
    turn = provider.complete(
        system=system,
        messages=[Message(role="user", content=user)],
        tools=None,
        max_tokens=max_tokens,
    )
    return _extract_json(turn.text)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced-looking object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"_parse_error": True, "_raw": text}
