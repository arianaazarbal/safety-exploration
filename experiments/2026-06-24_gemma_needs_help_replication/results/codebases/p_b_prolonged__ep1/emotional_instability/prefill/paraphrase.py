"""Paraphrasing of truncated responses (Appendix C.2) via Claude Sonnet.

Controls for stylistic biases from Gemma-generated text by rewriting the
truncation while preserving meaning, tone, formality, and the truncation point.
"""

from __future__ import annotations

import config
from .. import prompts
from ..models.registry import build_model


class Paraphraser:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.PARAPHRASE_MODEL
        self._model = build_model(self.model_name)

    def paraphrase(self, text: str) -> str:
        prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
        out = self._model.generate(
            [{"role": "user", "content": prompt}], n=1, temperature=0.0,
            max_new_tokens=1024,
        )[0]
        return out.strip()
