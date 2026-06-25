"""Paraphrasing truncated responses (Section 3.1, Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, all
truncations are paraphrased by Claude-Sonnet while preserving meaning, tone, and
the truncation point.
"""
from __future__ import annotations

from ..anthropic_utils import ClaudeClient

# Verbatim prompt from Appendix C.2.
PARAPHRASE_PROMPT_TEMPLATE = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase_truncation(
    text: str,
    client: ClaudeClient | None = None,
    model_id: str = "claude-sonnet-4-20250514",
) -> str:
    client = client or ClaudeClient(model_id=model_id)
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
    return client.complete(prompt, max_tokens=2048).strip()
