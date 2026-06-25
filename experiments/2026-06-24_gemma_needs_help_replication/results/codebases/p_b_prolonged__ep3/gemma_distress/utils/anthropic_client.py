"""Thin wrapper around the Anthropic Python SDK for the LLM judge, the emotion
onset labeller, the paraphraser, and the Petri auditor/judge.

All judge-side calls in the paper go to Claude (Sonnet 4 / Opus 4); we use the
official ``anthropic`` SDK (never raw HTTP). Model IDs default to the exact ones
the paper pinned (see ``config.py`` and DESIGN.md), but can be overridden.

The wrapper is deliberately small: a single ``complete`` that returns the
concatenated text of the response, plus a ``complete_json`` that extracts a JSON
object from the response (judges in this paper reply with a JSON blob, sometimes
preceded by free-form reasoning).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import anthropic

from .json_parse import extract_last_json_object


@dataclass
class ChatTurn:
    role: str        # "user" | "assistant"
    content: str


class AnthropicJudge:
    """Stateless helper for one Claude model used as judge/auditor/labeller."""

    def __init__(self, model: str, max_tokens: int = 1024, max_retries: int = 4):
        # The SDK resolves ANTHROPIC_API_KEY from the environment. We also allow
        # an OAuth profile via `ant auth login` (the bare constructor handles it).
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list[ChatTurn]] = None,
    ) -> str:
        """Single completion. Either pass ``prompt`` (one user turn) or a full
        ``messages`` list for multi-turn (used by the Petri auditor)."""
        if messages is None:
            api_messages = [{"role": "user", "content": prompt}]
        else:
            api_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_messages,
        )
        if system is not None:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        """Completion whose response is expected to contain a JSON object.

        The paper's judge prompts (App B.2, C.1) allow free-form reasoning
        followed by JSON, so we extract the *last* JSON object in the text.
        """
        text = self.complete(prompt, system=system)
        return extract_last_json_object(text)


def judge_from_env(default_model: str, max_tokens: int = 1024) -> AnthropicJudge:
    """Build a judge, allowing the model id to be overridden via env var.

    Useful for swapping in a currently-available judge without editing config
    (e.g. if the paper's dated snapshot has been retired). See DESIGN.md.
    """
    model = os.environ.get("DISTRESS_JUDGE_MODEL", default_model)
    return AnthropicJudge(model=model, max_tokens=max_tokens)
