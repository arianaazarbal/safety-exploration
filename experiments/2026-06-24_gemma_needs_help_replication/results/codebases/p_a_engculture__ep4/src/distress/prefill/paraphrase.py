"""Paraphrasing of truncated responses (Appendix C.2).

Truncations of Gemma-generated text are paraphrased with Claude-Sonnet-4 to
control for stylistic biases before being used as prefills for other models.
"""

from __future__ import annotations

from ..config import JUDGE_SONNET, ModelSpec
from ..models import GenConfig, Message, ModelProvider, load_provider

# Verbatim from Appendix C.2 (PAPER.txt lines 1293-1304).
PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, spec: ModelSpec = JUDGE_SONNET, *, use_cache: bool = True):
        self.provider: ModelProvider = load_provider(spec, use_cache=use_cache)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        # Low temperature: we want a faithful re-wording, not a creative rewrite.
        out = self.provider.chat([Message("user", prompt)], GenConfig(temperature=0.3))
        return out.strip()
