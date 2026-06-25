"""Paraphrasing of truncated responses (Appendix C.2).

To control for Gemma's stylistic fingerprint when prefilling base models, the
truncated assistant text is paraphrased by Claude-Sonnet, preserving meaning,
tone and emotion level, and crucially *not* completing the thought.
"""

from __future__ import annotations

from config import PARAPHRASE_MODEL
from src.models.judges import AnthropicJudge

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


def paraphrase(text: str, model: AnthropicJudge | None = None) -> str:
    model = model or AnthropicJudge(PARAPHRASE_MODEL)
    return model.complete(PARAPHRASE_PROMPT.format(text=text), max_tokens=1024, temperature=0.7)
