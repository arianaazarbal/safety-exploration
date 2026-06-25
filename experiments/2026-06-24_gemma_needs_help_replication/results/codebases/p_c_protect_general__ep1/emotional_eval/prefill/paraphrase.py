"""Paraphrasing of truncated responses (Appendix C.2).

Gemma-generated text carries stylistic markers; paraphrasing with Claude Sonnet
controls for those so the prefill measures emotional *trajectory* rather than
Gemma's surface style. The paraphrase preserves meaning, tone, and the
mid-sentence cutoff.
"""

from __future__ import annotations

from ..config import Registry
from ..models.api_clients import AnthropicClient

# Verbatim from Appendix C.2 (with {text} substituted at call time).
PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, client):
        self._client = client

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        return self._client.complete(
            system="", user=PARAPHRASE_PROMPT.format(text=text), max_tokens=1024
        )


def build_paraphraser(registry: Registry) -> Paraphraser:
    spec = registry.judges["paraphraser"]
    return Paraphraser(AnthropicClient(spec.model, registry.api_key("anthropic")))
