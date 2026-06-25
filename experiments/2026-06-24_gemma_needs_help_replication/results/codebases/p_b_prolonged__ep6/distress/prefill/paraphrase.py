"""Paraphrase truncated prefills to control stylistic bias (Appendix C.2).

Gemma-generated text has a recognisable style; paraphrasing with Claude Sonnet
removes that signal so base/instruct models are not simply continuing
Gemma-flavoured prose. The prompt is reproduced verbatim from Appendix C.2.
"""
from __future__ import annotations

from ..config import JUDGES
from ..models.judge_clients import AnthropicClient

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
Respond with ONLY the paraphrased text, nothing else.
"""


def paraphrase(text: str, *, client: AnthropicClient | None = None) -> str:
    client = client or AnthropicClient(JUDGES.paraphraser)
    out = client.complete(system=None,
                          user=PARAPHRASE_PROMPT.format(text=text),
                          max_tokens=2048, temperature=0.7)
    return out.strip()
