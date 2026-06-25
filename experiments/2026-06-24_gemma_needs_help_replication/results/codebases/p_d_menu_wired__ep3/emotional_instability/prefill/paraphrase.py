"""Paraphrase truncated responses (Appendix C.2) to control for the stylistic
bias of Gemma-generated text when prefilling other models.
"""
from __future__ import annotations

from ..models import ChatMessage, ModelClient
from ..prompts import PARAPHRASE_PROMPT


def paraphrase(client: ModelClient, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.replace("{text}", text)
    result = client.chat([ChatMessage("user", prompt)],
                         temperature=0.7, max_new_tokens=512)
    return result.text.strip()
