"""Paraphrasing of truncated responses (§3.1, Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased by Claude Sonnet 4 — preserving meaning, tone, and the (intentional)
mid-sentence ending — before being used as a prefill. Prompt reproduced verbatim
from Appendix C.2.
"""

from __future__ import annotations

from ..models.anthropic_client import AnthropicClient

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(client: AnthropicClient, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.replace("{text}", text)
    out = client.call_text(prompt, temperature=0.7, max_tokens=2048)
    return out.strip()
