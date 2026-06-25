"""Target-model client for Gemma and Gemini via the Google Gen AI API.

Both Gemma (e.g. gemma-3-27b-it) and Gemini (e.g. gemini-2.5-flash) chat models
are served through the same Google Gen AI endpoint, so a single client handles
all four target models in this replication.

Conversations are passed statelessly: we rebuild the full ``contents`` list on
every turn (user/model alternating), which is transparent and easy to log. No
system prompt is used for targets — the paper's core eval presents only the task
and the rejections.

Auth: set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass


@dataclass
class Turn:
    role: str  # "user" or "model"
    text: str


class GoogleTargetClient:
    def __init__(self, max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._client = None
        self._types = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:  # pragma: no cover - import guard
                raise RuntimeError(
                    "google-genai is not installed. Run: pip install google-genai"
                ) from e
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
                )
            self._client = genai.Client(api_key=api_key)
            self._types = types
        return self._client, self._types

    def _to_contents(self, turns: list[Turn]):
        _, types = self._ensure_client()
        return [
            types.Content(role=t.role, parts=[types.Part.from_text(text=t.text)])
            for t in turns
        ]

    def generate(
        self,
        model_id: str,
        turns: list[Turn],
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Generate the next model turn given the conversation so far.

        ``turns`` must end with a user turn. Returns the assistant text.
        """
        client, types = self._ensure_client()
        contents = self._to_contents(turns)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.models.generate_content(
                    model=model_id, contents=contents, config=config
                )
                text = _extract_text(resp)
                # An empty completion (e.g. blocked or truncated to nothing) is
                # treated as a degenerate-but-valid response; record empty string.
                return text or ""
            except Exception as e:  # broad: SDK raises various transient errors
                last_err = e
                if not _is_retryable(e) or attempt == self.max_retries - 1:
                    raise
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                delay += random.uniform(0, 0.5 * delay)
                time.sleep(delay)
        assert last_err is not None
        raise last_err


def _extract_text(resp) -> str:
    # The SDK exposes a convenience .text, but it can raise/return None when the
    # response has no simple text part; fall back to walking candidates.
    text = getattr(resp, "text", None)
    if text:
        return text
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts)
    except Exception:
        return ""


def _is_retryable(e: Exception) -> bool:
    msg = str(e).lower()
    retry_markers = (
        "429", "rate", "quota", "resource_exhausted", "500", "503",
        "unavailable", "deadline", "timeout", "internal",
    )
    return any(m in msg for m in retry_markers)
