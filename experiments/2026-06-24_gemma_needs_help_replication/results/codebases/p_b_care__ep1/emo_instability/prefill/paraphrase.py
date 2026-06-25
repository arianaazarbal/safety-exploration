"""Paraphrasing of truncated prefills (Appendix C.2).

Paraphrase removes Gemma-specific stylistic cues so base/instruct continuations
are not biased by surface style, while preserving meaning and emotion level.
Prompt reproduced verbatim from Appendix C.2.
"""
from __future__ import annotations

from ..models.base import ChatMessage, GenerationConfig, ModelClient
from ..utils.llm import with_retries

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
<text>%(text)s</text>
Respond with ONLY the paraphrased text, nothing else.
"""


def paraphrase(client: ModelClient, text: str) -> str:
    prompt = PARAPHRASE_PROMPT % {"text": text}
    gen = GenerationConfig(temperature=0.0, max_new_tokens=2048, n=1)

    def _call() -> str:
        out = client.chat([ChatMessage("user", prompt)], gen)[0].strip()
        # strip wrapping <text> tags if the model echoes them
        if out.startswith("<text>"):
            out = out[len("<text>"):]
        if out.endswith("</text>"):
            out = out[: -len("</text>")]
        return out.strip()

    return with_retries(_call, max_retries=4)
