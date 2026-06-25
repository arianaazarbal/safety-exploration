"""Paraphrase truncated prefills to control for Gemma's stylistic fingerprints
(Section 3.1, Appendix C.2).

Both base and instruct models continue from the *same* paraphrased prefill, so
any difference in continuation emotion is attributable to the model rather than
to surface features of Gemma's original phrasing.
"""

from __future__ import annotations

from emotional_stability.config import PARAPHRASE_MODEL, Settings
from emotional_stability.models.anthropic_client import AnthropicClient
from emotional_stability.prompts.prefill import build_paraphrase_prompt
from emotional_stability.records import Message


class Paraphraser:
    def __init__(self, model: str = PARAPHRASE_MODEL, settings: Settings | None = None):
        self._client = AnthropicClient(model, settings=settings)

    def paraphrase(self, text: str) -> str:
        reply = self._client.complete(
            [Message(role="user", content=build_paraphrase_prompt(text))],
            temperature=0.0,
            max_tokens=2048,
        )
        return reply.strip()
