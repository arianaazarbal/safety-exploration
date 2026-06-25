"""Paraphrasing of truncated prefills (Appendix C.2).

Truncations are paraphrased with Claude Sonnet to control for stylistic biases
from using Gemma-generated text as the prefill (so base/instruct comparisons are
not confounded by Gemma's idiosyncratic phrasing).
"""
from __future__ import annotations

from ..backends.anthropic_client import complete
from ..config import JUDGE_MODEL

# Verbatim from Appendix C.2.
PARAPHRASE_PROMPT_TEMPLATE = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, model: str = JUDGE_MODEL) -> str:
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
    out = complete(model=model, system=None,
                   messages=[{"role": "user", "content": prompt}],
                   max_tokens=1024)
    return out.strip()
