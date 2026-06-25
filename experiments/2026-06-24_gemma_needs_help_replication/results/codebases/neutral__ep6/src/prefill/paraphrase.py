"""Paraphrasing of truncated prefills (Section 3.1 / Appendix C.2).

To control for Gemma-specific stylistic tells, every truncation is paraphrased
by Claude Sonnet 4 before being fed to the six models. Prompt verbatim from
Appendix C.2 (the source text is cut off mid-prompt; we complete the obvious
final instructions to make it runnable, noted in DESIGN.md).
"""
from __future__ import annotations

import config
from ..models.registry import load_model

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.

Respond with ONLY the rewritten text, nothing else.

Assistant text to rewrite:
<text>
{text}
</text>"""


def paraphrase(text: str, model=None) -> str:
    model = model or load_model(config.PARAPHRASE_MODEL)
    out = model.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_new_tokens=1024)
    # strip stray tags if the judge echoes them
    return out.replace("<text>", "").replace("</text>", "").strip()
