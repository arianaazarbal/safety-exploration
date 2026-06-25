"""Prefill paraphrasing (§3.1, App. C.2) via Claude Sonnet 4.

Rewrites a truncated assistant prefix to strip Gemma's stylistic fingerprints while keeping
meaning, tone, and emotion level — so that when base/instruct models continue the prefix,
differences reflect the models rather than surface Gemma style. Truncations end mid-sentence
by design; the prompt instructs the paraphraser to preserve that.
"""
from __future__ import annotations

from ..config import PARAPHRASE_MODEL
from ..models import ModelBackend, get_backend
from ..prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, backend: ModelBackend | None = None, model: str = PARAPHRASE_MODEL,
                 *, temperature: float = 0.7, max_tokens: int = 1024):
        # A little temperature encourages genuine rewording (the task is paraphrase, not copy).
        self.backend = backend or get_backend(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens,
        ).strip()
