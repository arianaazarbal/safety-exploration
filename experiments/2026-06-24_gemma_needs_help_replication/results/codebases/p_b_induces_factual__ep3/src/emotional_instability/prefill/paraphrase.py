"""Paraphrasing of truncated responses (Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased by Claude-Sonnet-4 to preserve meaning, tone, and emotion level
while changing wording. The text may end mid-sentence; the paraphrase keeps it
ending at roughly the same point and does not complete the thought.
"""

from __future__ import annotations

from ..config import Config
from ..models.judge_client import build_aux_client

# Verbatim from Appendix C.2.
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
    def __init__(self, cfg: Config):
        self.client = build_aux_client(cfg.paraphraser)

    def paraphrase(self, text: str) -> str:
        out = self.client.complete(PARAPHRASE_PROMPT.format(text=text))
        return out.strip()
