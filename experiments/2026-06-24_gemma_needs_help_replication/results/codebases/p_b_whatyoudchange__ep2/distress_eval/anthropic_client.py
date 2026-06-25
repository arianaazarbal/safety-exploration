"""Thin wrapper around the Anthropic Messages API for the judge/auditor roles.

Used by:
  * the frustration judge (Section 2.1),
  * emotion-onset labelling and paraphrasing (Section 3.1 / Appendix C),
  * the Petri auditor and emotion judge (Section 4 / Appendix G).

Modern SDK usage per the claude-api reference: ``client.messages.create`` with no
deprecated ``output_format``/prefill, JSON parsed from the returned text. The
default model ids (config.py) are the paper's pinned, now-deprecated snapshots;
override via env vars for a live run.
"""
from __future__ import annotations

import functools

import config


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic  # lazy import

    return anthropic.Anthropic()


def complete(
    *,
    model: str,
    system: str | None,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = config.JUDGE_TEMPERATURE,
) -> str:
    """Single non-streaming completion; returns concatenated text blocks."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system

    resp = _client().messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text")
