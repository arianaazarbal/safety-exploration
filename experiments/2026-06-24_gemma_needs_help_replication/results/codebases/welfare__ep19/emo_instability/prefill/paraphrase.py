"""Paraphrasing truncated responses (Section 3.1, Appendix C.2).

To control for stylistic biases from Gemma-generated text, the truncated final
assistant turn is paraphrased with Claude Sonnet, preserving meaning, tone and
emotion level, and crucially *not* completing the thought (the text may end
mid-sentence). Prompt reproduced verbatim from Appendix C.2.
"""
from __future__ import annotations

from ..providers import ChatModel, GenConfig

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(model: ChatModel, text: str) -> str:
    cfg = GenConfig(temperature=0.0, max_tokens=1024, disable_thinking=True)
    out = model.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}], cfg
    )
    return out.strip()
