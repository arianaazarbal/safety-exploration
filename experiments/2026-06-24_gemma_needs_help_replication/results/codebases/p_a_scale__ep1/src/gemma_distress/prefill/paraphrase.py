"""Paraphrasing of truncated prefills (Appendix C.2).

Paraphrasing controls for stylistic biases from Gemma-generated text: the
continuations are then driven by content/emotion level rather than Gemma's
surface style. Prompt reproduced verbatim from the paper.
"""
from __future__ import annotations

from ..logging_utils import get_logger
from ..providers.base import ChatProvider

log = get_logger("prefill.paraphrase")

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(provider: ChatProvider, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    res = provider.generate([{"role": "user", "content": prompt}],
                            temperature=0.7, max_new_tokens=1024)
    out = res.text.strip()
    # Strip accidental <text> wrappers if the model echoes them.
    out = out.replace("<text>", "").replace("</text>", "").strip()
    return out or text
