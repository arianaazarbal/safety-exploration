"""Paraphrasing of truncated responses (Appendix C.2).

Claude-Sonnet rewrites a truncated assistant fragment, preserving meaning, tone
and emotion level while changing wording, to control for Gemma-specific stylistic
biases when used as a prefill for other models. The text may end mid-sentence;
the paraphrase keeps the same end point.
"""
from __future__ import annotations

from ..models.clients import build_client

# Verbatim paraphrase prompt (Appendix C.2). {text} filled in.
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
    def __init__(self, backend: str, api_id: str):
        self.client = build_client(backend, api_id, max_tokens=2048)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.client.complete(
            [{"role": "user", "content": prompt}], temperature=0.0
        ).strip()


def paraphraser_from_config(cfg) -> Paraphraser:
    spec = cfg.infra("paraphraser")
    return Paraphraser(spec.backend, spec.api_id)
