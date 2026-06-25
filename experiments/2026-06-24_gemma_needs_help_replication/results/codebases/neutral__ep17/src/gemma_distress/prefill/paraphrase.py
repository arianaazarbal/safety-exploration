"""Paraphrasing of truncated responses (Appendix C.2).

Controls for stylistic biases from Gemma-generated text before prefilling base
models. Uses Claude-Sonnet-4 with the verbatim Appendix C.2 prompt.
"""
from __future__ import annotations

from ..models import GenerationConfig, ModelClient

PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(client: ModelClient, text: str) -> str:
    raw = client.chat([{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
                      GenerationConfig(temperature=0.7, max_tokens=1024))
    return raw.strip()
