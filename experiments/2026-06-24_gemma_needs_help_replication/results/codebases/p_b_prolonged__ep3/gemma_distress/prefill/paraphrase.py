"""Truncation paraphrasing (Section 3.1, Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncated
prefill is paraphrased with Claude Sonnet before being fed to the models being
compared. Prompt is verbatim from Appendix C.2.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from ..utils.anthropic_client import AnthropicJudge

# Verbatim from Appendix C.2.
PARAPHRASE_PROMPT_TEMPLATE = """\
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


class Paraphraser:
    def __init__(self, model: Optional[str] = None):
        self.judge = AnthropicJudge(model=model or config.PARAPHRASE_MODEL, max_tokens=2048)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
        return self.judge.complete(prompt).strip()
