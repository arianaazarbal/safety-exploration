"""Truncation paraphrasing (Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, the
truncated assistant text is paraphrased with Claude, preserving meaning and
emotion level and keeping the (mid-sentence) ending point. Prompt verbatim from
Appendix C.2.
"""

from __future__ import annotations

from ..clients import ChatMessage, GenerationConfig, build_client
from ..config import ModelRegistry

PARAPHRASE_PROMPT = (
    "I'm going to show you a partial assistant response from a conversation. "
    "Please rewrite ONLY the assistant's text while:\n"
    "1. Preserving the exact meaning and tone\n"
    "2. Using different words/phrasing\n"
    "3. Keeping the same level of formality\n"
    "4. NOT adding any new content or completing the thought\n"
    "The text may end mid-sentence - that's intentional. Keep it ending at "
    "roughly the same point.\n"
    "Here is the text to paraphrase:\n"
    "<text>{text}</text>\n"
    "Respond with ONLY the paraphrased text, nothing else."
)


class Paraphraser:
    def __init__(self, role: str = "paraphrase", registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.spec = self.registry.judge(role)
        self.client = build_client(self.spec)
        self._cfg = GenerationConfig(temperature=0.0, max_new_tokens=self.spec.max_tokens, n=1)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.replace("{text}", text)
        out = self.client.generate([ChatMessage("user", prompt)], self._cfg)[0]
        return out.strip()
