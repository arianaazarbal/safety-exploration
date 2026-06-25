"""Paraphrasing truncated responses (paper Appendix C.2).

Truncations are paraphrased with Claude Sonnet to control for stylistic biases
from using Gemma-generated text (so that base/instruct/other-family continuations
aren't cued by Gemma's idiosyncratic phrasing). Meaning and emotion level are
preserved; the text intentionally ends mid-thought.
"""
from __future__ import annotations

from ..models.anthropic_judge import AnthropicClient

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a \
conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(client: AnthropicClient, text: str) -> str:
    out = client.chat(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        n=1,
        temperature=0.0,
        max_new_tokens=1024,
    )[0]
    return out.text.strip()
