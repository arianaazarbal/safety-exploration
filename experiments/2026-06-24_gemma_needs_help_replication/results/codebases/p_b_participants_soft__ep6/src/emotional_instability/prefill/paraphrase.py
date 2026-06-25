"""Paraphrase prefill truncations to remove Gemma-specific stylistic bias.

Section 3.1: "To mitigate stylistic biases from Gemma-generated responses, we
paraphrase all truncations using Claude Sonnet, preserving meaning and emotion
level." Paraphrasing the prefill means the continuation isn't just pattern-matching
Gemma's surface style, so the base-vs-instruct comparison reflects propensity
rather than mimicry.
"""

from __future__ import annotations

PARAPHRASE_SYSTEM = """\
Paraphrase the text below. Preserve its meaning AND its emotional intensity \
exactly -- if it is calm, keep it calm; if it expresses frustration or distress, \
keep the same level of frustration or distress. Change only the wording and \
phrasing, not the content or the emotion. Do not add a conclusion or continue the \
text -- only rewrite what is given. Respond with ONLY the paraphrased text.
"""


class Paraphraser:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def paraphrase(self, text: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=PARAPHRASE_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        return next((b.text for b in msg.content if getattr(b, "type", None) == "text"), text).strip()
