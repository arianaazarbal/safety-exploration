"""Anthropic API client used for judges, the Petri auditor, the onset labeller,
and the paraphraser.

Uses the official `anthropic` SDK. Streaming is used for safety on longer
outputs; we collect the final message and return its text. The judge models are
pinned to the paper's dated IDs by default (see config.models) -- this is a
faithfulness replication where the autorater identity is part of the instrument.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationResult


class AnthropicClient:
    supports_prefill = False

    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        default_temperature: float = 0.0,
        default_max_new_tokens: int = 1024,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        self.key = key
        self.model_id = model_id
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self._api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ[self._api_key_env])
        return self._client

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
    def _one(
        self,
        messages: list[dict],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._ensure_client()
        kwargs = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            final = stream.get_final_message()
        return "".join(b.text for b in final.content if b.type == "text")

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        n: int = 1,
        system: Optional[str] = None,
    ) -> list[GenerationResult]:
        # Separate an optional leading system message from the turn list.
        sys_text = system
        turns = []
        for m in messages:
            if m.role == "system" and sys_text is None:
                sys_text = m.content
            else:
                turns.append({"role": m.role, "content": m.content})
        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_new_tokens or self.default_max_new_tokens
        return [
            GenerationResult(text=self._one(turns, sys_text, temp, max_tok))
            for _ in range(n)
        ]

    def generate_prefill(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Anthropic client is used only as a judge/auditor.")


# --------------------------------------------------------------------------- #
# JSON extraction shared by judges (the prompts ask for trailing JSON).
# --------------------------------------------------------------------------- #
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_last_json(text: str) -> Optional[dict]:
    """Parse the last JSON object in the model's response (judges emit one)."""
    matches = list(_JSON_OBJ_RE.finditer(text))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    # Fall back to a greedy outermost-brace parse.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
