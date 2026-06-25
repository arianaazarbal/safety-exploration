"""Paraphrase truncated assistant text with Claude (paper Appendix C.2).

Used to control for stylistic biases from Gemma-generated prefills before they
are fed to other models. Prompt reproduced verbatim from Appendix C.2.
"""

from __future__ import annotations

from emo.config import PARAPHRASE_MODEL
from emo.judges.anthropic_client import complete

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


def paraphrase(text: str, model: str = PARAPHRASE_MODEL) -> str:
    if not text.strip():
        return text
    out = complete(model, user=PARAPHRASE_PROMPT.format(text=text), max_tokens=1024)
    # Strip any stray <text> tags the model might echo.
    return out.replace("<text>", "").replace("</text>", "").strip()
