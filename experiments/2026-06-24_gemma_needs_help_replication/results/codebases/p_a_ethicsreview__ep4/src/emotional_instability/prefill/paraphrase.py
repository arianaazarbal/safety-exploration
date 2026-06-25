"""Paraphrasing of truncated responses (Section 3.1 / Appendix C.2).

Truncations are paraphrased with Claude Sonnet to control for Gemma's stylistic
fingerprint, so that base/instruct continuations are compared from neutral
starting text. The prompt is reproduced verbatim from Appendix C.2.
"""

from __future__ import annotations

from ..models.anthropic_backend import AnthropicClient

PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the \
same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, model_id: str, max_tokens: int = 1024):
        self._client = AnthropicClient(model_id)
        self._max_tokens = max_tokens

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self._client.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=self._max_tokens,
        )
        return out.strip()
