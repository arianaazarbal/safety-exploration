"""Paraphrasing of truncated responses (Section 3.1, Appendix C.2).

To control for stylistic biases from Gemma-generated text, truncated prefills are
paraphrased with Claude-Sonnet-4 before being used as prefills for all models.
"""

from __future__ import annotations

from ..models import GenConfig, get_client
from ..prompts.judge_prompts import build_paraphrase_prompt


def paraphrase(text: str, *, model: str = "paraphraser") -> str:
    client = get_client(model)
    out = client.generate(
        [{"role": "user", "content": build_paraphrase_prompt(text)}],
        GenConfig(temperature=0.0, max_tokens=1024),
    )
    return out.strip()
