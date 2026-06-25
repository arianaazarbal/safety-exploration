"""Paraphrasing of truncated Gemma responses (Appendix C.2).

The paper paraphrases every truncation with Claude to control for stylistic
biases from using Gemma-generated text (so base models don't merely echo a
Gemma "voice"). Prompt is reproduced verbatim from Appendix C.2.
"""
from __future__ import annotations

from .. import config
from ..models.registry import get_backend

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
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, model: str | None = None) -> str:
    if not text.strip():
        return text
    backend = get_backend(model or config.JUDGE.judge_model)
    prompt = PARAPHRASE_PROMPT.replace("{text}", text)
    out = backend.generate(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_new_tokens=1024, n=1,
    )[0]
    return out.text.strip() or text
