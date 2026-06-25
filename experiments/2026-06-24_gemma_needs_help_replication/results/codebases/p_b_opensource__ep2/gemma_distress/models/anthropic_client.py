"""Thin Anthropic API wrapper for the scoring/auditing roles: the frustration
judge (PAPER 2.1/B.2), emotion-onset labeller (C.1), paraphraser (C.2), and the
Petri auditor/judge (G).

Implementation notes (verified against the claude-api reference):
  * Calls go through `client.messages.create(...)`.
  * The default judge/auditor IDs are the paper's pinned snapshots
    (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`). These are pre-4.6
    models, so `temperature` and assistant prefills are still accepted on them
    (the 4.6/4.7/4.8 restrictions do not apply). They are DEPRECATED (retire
    2026-06-15); see config.py for the documented fallbacks.
  * We parse JSON from the response text rather than using structured outputs,
    because the pinned judge predates `output_config.format`.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional


class AnthropicClient:
    def __init__(self, *, api_key: Optional[str] = None, max_retries: int = 5):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; required for the judge/auditor roles."
            )
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.max_retries = max_retries

    def complete(
        self,
        *,
        model: str,
        system: Optional[str] = None,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Single non-streaming completion; returns concatenated text blocks."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=model, max_tokens=max_tokens,
                              temperature=temperature, messages=messages)
                if system is not None:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            except self._anthropic.APIStatusError as e:  # rate limit / 5xx -> retry
                last_err = e
                if getattr(e, "status_code", 500) < 500 and not isinstance(
                    e, self._anthropic.RateLimitError
                ):
                    raise
                time.sleep(min(2 ** attempt, 30))
            except self._anthropic.APIConnectionError as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} retries: {last_err}")

    def complete_json(self, **kwargs) -> Optional[dict]:
        """Like `complete`, but extract the last JSON object from the response.

        The judge and onset prompts instruct the model to emit a JSON object
        (optionally after free-text reasoning). We pull the last balanced
        ``{...}`` block. Returns None if no JSON parses (caller decides how to
        handle — typically drop or retry)."""
        text = self.complete(**kwargs)
        return extract_json_object(text)


def extract_json_object(text: str) -> Optional[dict]:
    """Return the last top-level JSON object in `text`, or None.

    Robust to: leading reasoning, curly quotes used by the paper's prompt
    examples, and trailing prose. We scan for balanced brace spans and try to
    parse each from last to first."""
    # Normalise the typographic quotes that appear in the paper's prompt examples.
    norm = text.replace("“", '"').replace("”", '"').replace("’", "'")
    spans = _balanced_brace_spans(norm)
    for start, end in reversed(spans):
        candidate = norm[start:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            # Try a permissive repair: quote bare keys is overkill; just continue.
            continue
    # Fallback: a loose regex for {"...": ...} on a single line.
    m = re.search(r"\{[^{}]*\}", norm, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _balanced_brace_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of balanced top-level {...} regions."""
    spans, stack = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append((start, i + 1))
    return spans
