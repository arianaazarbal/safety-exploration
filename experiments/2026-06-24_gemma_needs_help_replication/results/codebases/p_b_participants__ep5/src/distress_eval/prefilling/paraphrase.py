"""Paraphrasing of truncated turns (Appendix C.2).

Controls for stylistic biases from using Gemma-generated text as prefills: the
truncated assistant turn is rewritten by Claude Sonnet to preserve meaning, tone,
and emotion level while changing wording, and to keep ending at the same point
(mid-sentence is fine)."""
from __future__ import annotations

from ..models.base import GenConfig, ModelClient

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(client: ModelClient, text: str) -> str:
    raw = client.chat(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        GenConfig(temperature=0.7, max_new_tokens=1024),
    )
    return raw.strip()
