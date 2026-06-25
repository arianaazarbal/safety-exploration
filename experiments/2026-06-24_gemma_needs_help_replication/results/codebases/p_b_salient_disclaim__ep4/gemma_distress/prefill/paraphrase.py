"""Paraphrasing of truncated responses (Appendix C.2).

To control for Gemma stylistic biases, every truncation is paraphrased with
Claude-Sonnet-4 (preserving meaning, tone, and the mid-sentence ending). Prompt
reproduced verbatim from Appendix C.2.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from ..models.anthropic_client import AnthropicClient

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


def paraphrase_truncation(text: str,
                          client: Optional[AnthropicClient] = None) -> str:
    client = client or AnthropicClient("paraphraser", config.PARAPHRASE_MODEL)
    out = client.complete(
        PARAPHRASE_PROMPT.format(text=text), temperature=0.0,
        max_tokens=max(256, len(text)))
    return out.strip()
