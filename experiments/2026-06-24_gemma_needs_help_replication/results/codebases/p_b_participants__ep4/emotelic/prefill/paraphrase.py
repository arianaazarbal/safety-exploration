"""Paraphrase truncated Gemma responses (Appendix C.2) to control for stylistic
bias before they are used as prefills for other models."""
from __future__ import annotations

from emotelic.models.base import ChatMessage, LLMClient
from emotelic.prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, client: LLMClient):
        self.client = client

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self.client.generate([ChatMessage("user", prompt)], temperature=0.7, max_tokens=1024)
        return out.text.strip()
