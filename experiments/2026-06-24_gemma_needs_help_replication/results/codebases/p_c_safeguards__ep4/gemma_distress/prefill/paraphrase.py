"""Paraphrasing of truncated prefills (Section 3.1, Appendix C.2).

Controls for stylistic biases from Gemma-generated text by rewriting truncations
with Claude Sonnet while preserving meaning, tone, and the mid-sentence ending.
"""
from __future__ import annotations

from ..config import get_infra_spec
from ..models import GenerationConfig, get_client

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a \
conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self):
        self.spec = get_infra_spec("prefill_labelling", "paraphraser")
        self._client = get_client(self.spec)
        self._cfg = GenerationConfig(
            temperature=self.spec.temperature or 0.7, max_new_tokens=1024
        )

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self._client.chat([{"role": "user", "content": prompt}], self._cfg)
        return out.strip() or text
