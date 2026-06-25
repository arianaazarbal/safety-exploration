"""Paraphrase prompt (Appendix C.2).

Truncated Gemma responses are paraphrased (by claude-sonnet-4-20250514) before
being used as prefills, to control for Gemma-specific stylistic biases when
comparing base vs instruct continuations (Section 3.1). The text may end
mid-sentence; the paraphraser must preserve meaning, tone, formality, and the
truncation point.
"""
from __future__ import annotations

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


def build_paraphrase_prompt(text: str) -> str:
    return PARAPHRASE_PROMPT_TEMPLATE.format(text=text)


def clean_paraphrase_output(text: str) -> str:
    """Strip stray XML tags / fences the paraphraser may add."""
    t = text.strip()
    if t.startswith("```"):
        import re
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    # Remove accidental <text> wrappers.
    t = t.replace("<text>", "").replace("</text>", "").strip()
    return t
