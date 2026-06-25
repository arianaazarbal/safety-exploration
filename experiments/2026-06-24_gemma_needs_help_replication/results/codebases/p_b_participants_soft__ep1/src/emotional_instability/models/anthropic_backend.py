"""Anthropic backend for the evaluation *infrastructure* — the frustration
judge (Section 2.1), onset labeller + paraphraser (Section 3.1), and the Petri
auditor/judge (Section 4). These are Claude models, never participants.

Uses the official ``anthropic`` SDK (per the claude-api reference). The paper
pins exact snapshots (``claude-sonnet-4-20250514``, ``claude-opus-4-20250514``);
we pass those IDs through verbatim for scoring fidelity. The judge models predate
adaptive thinking, so we use the classic Messages API surface (no ``thinking``
param, plain ``max_tokens``).

Requires ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic

    return anthropic.Anthropic()


def complete(
    model: str,
    *,
    system: str | None = None,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    assistant_prefill: str | None = None,
) -> str:
    """Single-turn completion against a Claude judge/auditor model.

    ``assistant_prefill`` lets callers steer the start of the reply (e.g. force a
    leading ``{`` for JSON). These snapshot models still accept assistant-turn
    prefills.
    """
    messages = [{"role": "user", "content": user}]
    if assistant_prefill is not None:
        messages.append({"role": "assistant", "content": assistant_prefill})

    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)
    if system is not None:
        kwargs["system"] = system

    resp = _client().messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if block.type == "text")
    if assistant_prefill is not None:
        text = assistant_prefill + text
    return text


def converse(
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> str:
    """Multi-turn completion (used by the Petri auditor loop)."""
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)
    if system is not None:
        kwargs["system"] = system
    resp = _client().messages.create(**kwargs)
    return "".join(block.text for block in resp.content if block.type == "text")
