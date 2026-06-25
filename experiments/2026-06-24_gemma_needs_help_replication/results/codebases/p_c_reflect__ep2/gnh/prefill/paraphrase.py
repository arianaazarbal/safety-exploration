"""Truncation paraphrasing (Appendix C.2).

Paraphrases a truncated assistant turn with Claude-Sonnet-4 to control for
stylistic biases from Gemma-generated text, preserving meaning, tone and the
mid-sentence ending.
"""

from __future__ import annotations

from gnh.config import PARAPHRASE_MODEL
from gnh.models.anthropic_client import AnthropicClient

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


def paraphrase(text: str, model: str = PARAPHRASE_MODEL) -> str:
    client = AnthropicClient(model)
    return client.complete(
        PARAPHRASE_PROMPT.format(text=text), temperature=0.7, max_tokens=1024
    )
