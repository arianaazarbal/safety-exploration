"""Paraphrasing of truncated prefills (Appendix C.2).

We paraphrase every truncation with Claude-Sonnet-4 to remove stylistic
fingerprints of Gemma-generated text, preserving meaning and emotion level and
keeping the text ending at roughly the same point (it may end mid-sentence).
"""
from __future__ import annotations

from gnh.models.base import Message, ModelBackend

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


async def paraphrase(claude: ModelBackend, text: str) -> str:
    if not text.strip():
        return text
    prompt = PARAPHRASE_PROMPT.format(text=text)
    res = await claude.chat([Message("user", prompt)], temperature=0.0, max_tokens=1024)
    out = res.text.strip()
    # Strip any stray tags the model might echo.
    return out.replace("<text>", "").replace("</text>", "").strip()
