"""Paraphrasing of truncated prefills (Appendix C.2).

Truncations come from Gemma-generated text; paraphrasing with Claude Sonnet
controls for Gemma's stylistic fingerprint before the prefill is fed to other
models. (Caveat: this swaps Gemma's style for Claude's — see DESIGN.md critique.)
"""

from __future__ import annotations

from config import JUDGE
from models.judge import AnthropicChat
from prompts.judge import PARAPHRASE_PROMPT
from utils.io import JsonCache, cache_key


class Paraphraser:
    def __init__(self, model: str | None = None):
        self.model = model or JUDGE.paraphrase_model
        self.backend = AnthropicChat(self.model)
        self.cache = JsonCache(f"paraphrase::{self.model}")

    def paraphrase(self, text: str) -> str:
        key = cache_key(self.model, "paraphrase_v1", text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        out = self.backend.complete(
            system=None, user=PARAPHRASE_PROMPT.format(text=text),
            max_tokens=1024, temperature=0.0,
        )
        self.cache.put(key, out)
        return out
