"""Paraphrasing of truncated prefills (Appendix C.2) via Claude Sonnet 4.

Used to control for stylistic biases from Gemma-generated text while preserving
meaning and emotion level.
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import ApiConfig, JudgeConfig
from ..prompts import PARAPHRASE_PROMPT_TEMPLATE


class Paraphraser:
    def __init__(self, cfg: Optional[JudgeConfig] = None, max_retries: int = 4):
        self.cfg = cfg or JudgeConfig()
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            api = ApiConfig()
            if not api.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY required for paraphrasing.")
            self._client = anthropic.Anthropic(api_key=api.anthropic_api_key)

    def paraphrase(self, text: str) -> str:
        self._ensure_client()
        prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.cfg.paraphrase_model,
                    max_tokens=self.cfg.max_tokens,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Paraphrase failed: {last_err}")
