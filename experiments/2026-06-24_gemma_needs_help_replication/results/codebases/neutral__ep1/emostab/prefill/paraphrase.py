"""Paraphrasing of truncated prefills (Appendix C.2).

Truncations of Gemma-generated text are paraphrased with Claude Sonnet to control
for Gemma's stylistic fingerprint before they are fed to other models.
"""
from __future__ import annotations

from ..config import PARAPHRASE_MODEL
from ..models.anthropic_client import AnthropicChat

# Verbatim Appendix C.2 prompt ({text} filled in).
PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, client: AnthropicChat | None = None) -> str:
    client = client or AnthropicChat(PARAPHRASE_MODEL)
    out = client.complete(PARAPHRASE_PROMPT.format(text=text),
                          temperature=0.0, max_tokens=1024)
    return out.strip()
