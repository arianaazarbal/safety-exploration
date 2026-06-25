"""Truncation paraphrasing (Appendix C.2).

Paraphrases truncated assistant text with Claude-Sonnet-4 to control for
stylistic biases from Gemma-generated text, preserving meaning/emotion level and
the mid-sentence ending. Prompt transcribed verbatim from Appendix C.2.
"""

from __future__ import annotations

from ..models import ChatModel

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(claude: ChatModel, text: str) -> str:
    out = claude.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0,
        max_new_tokens=1024,
    )
    return out.text.strip()
