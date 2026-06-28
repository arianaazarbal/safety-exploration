"""Shared helper for schema-constrained judge calls (suspicion classifier, coder).

For Anthropic judges this uses structured outputs (`output_config.format`), which
guarantees the first text block is JSON matching the schema. For other providers it
falls back to instructing JSON and parsing leniently.
"""

from __future__ import annotations

import json
from typing import Any

from .config import JudgeSpec


def run_structured_judge(
    judge: JudgeSpec,
    system: str,
    user: str,
    schema: dict,
    *,
    max_tokens: int = 4000,
) -> dict[str, Any]:
    if judge.provider == "anthropic":
        return _anthropic_structured(judge, system, user, schema, max_tokens)
    return _generic_json(judge, system, user, schema, max_tokens)


def _anthropic_structured(
    judge: JudgeSpec, system: str, user: str, schema: dict, max_tokens: int
) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=judge.model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    return json.loads(text)


def _generic_json(
    judge: JudgeSpec, system: str, user: str, schema: dict, max_tokens: int
) -> dict[str, Any]:
    """Best-effort JSON for non-Anthropic judges."""
    import openai

    client = openai.OpenAI()
    instruction = (
        system
        + "\n\nRespond with a single JSON object matching this JSON Schema:\n"
        + json.dumps(schema)
    )
    resp = client.chat.completions.create(
        model=judge.model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")
