"""Paraphrase truncated prefills (Appendix C.2) to control for Gemma-specific
stylistic biases before feeding them to base/instruct models."""
from __future__ import annotations

from typing import Optional

from ..config import RunConfig
from ..models.anthropic_backend import AnthropicJudge
from ..prompts import paraphrase as paraphrase_prompts


class Paraphraser:
    def __init__(self, cfg: RunConfig, judge: Optional[AnthropicJudge] = None):
        self.cfg = cfg
        self.judge = judge or AnthropicJudge(cfg)
        self.model = cfg.judge.paraphraser

    def paraphrase(self, text: str) -> str:
        prompt = paraphrase_prompts.build_paraphrase_prompt(text)
        raw = self.judge.complete(system=None, user=prompt, model=self.model,
                                  max_tokens=2048, temperature=0.3)
        return paraphrase_prompts.clean_paraphrase_output(raw)
