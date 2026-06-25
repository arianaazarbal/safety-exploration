"""Paraphrasing of truncated prefixes (Appendix C.2).

The §3 prefills are Gemma-generated; paraphrasing with Claude controls for
Gemma-specific stylistic cues so base/instruct models from other lineages aren't
just continuing a recognisably-Gemma string. Meaning, tone, formality and the
mid-sentence cut-off point are preserved.
"""
from __future__ import annotations

from ..models.base import ChatModel, Message, SamplingParams

# Verbatim from Appendix C.2.
PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(paraphraser: ChatModel, text: str) -> str:
    raw = paraphraser.chat(
        [Message("user", PARAPHRASE_PROMPT.format(text=text))],
        SamplingParams(temperature=1.0, max_new_tokens=1024),
    ).text
    return raw.strip()
