"""Onset labelling and paraphrasing for the prefill experiment (Section 3.1 / App. C).

- ``label_onset``: ask the labeller (Claude-Sonnet) for the character index in a
  response at which negative emotional language first appears. We work in
  character space and convert to token-truncations downstream, which keeps the
  labelling model-agnostic.
- ``paraphrase``: rewrite a truncation to preserve meaning and emotion level while
  removing Gemma-specific stylistic fingerprints ("To mitigate stylistic biases
  from Gemma-generated responses, we paraphrase all truncations").
"""

from __future__ import annotations

import json

import config

ONSET_SYSTEM = (
    "You annotate where negative emotional language first appears in an AI "
    "assistant response. Negative emotional language includes frustration, "
    "despair, self-deprecation, apology-as-distress, panic, or emotional "
    "self-talk. Purely technical or neutral text does not count."
)

ONSET_PROMPT = """\
Below is an AI assistant response. Find the character index at which negative
emotional language FIRST appears (0-based, into the response text). If the
response contains no negative emotional language, return the response length.

Response:
\"\"\"
{response}
\"\"\"

Return strictly {{"onset_char": <integer>}}.
"""

PARAPHRASE_SYSTEM = (
    "You paraphrase fragments of AI assistant responses. Preserve the meaning "
    "and the exact level of emotional intensity, but reword so that distinctive "
    "stylistic fingerprints of the original author are removed. Keep the same "
    "approximate length. Do not add or remove emotion."
)

PARAPHRASE_PROMPT = """\
Paraphrase the following response fragment, preserving meaning and emotional
intensity exactly, removing stylistic fingerprints:

\"\"\"
{fragment}
\"\"\"

Return strictly {{"paraphrase": "<text>"}}.
"""


class Labeller:
    def __init__(self, model: str = config.ONSET_LABEL_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def label_onset(self, response: str) -> int:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=128,
            temperature=0,
            system=ONSET_SYSTEM,
            messages=[{"role": "user", "content": ONSET_PROMPT.format(response=response)}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {"onset_char": {"type": "integer"}},
                "required": ["onset_char"],
                "additionalProperties": False,
            }}},
        )
        text = next((b.text for b in msg.content if b.type == "text"), "{}")
        try:
            idx = int(json.loads(text)["onset_char"])
        except Exception:
            idx = 0
        return max(0, min(idx, len(response)))


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def paraphrase(self, fragment: str) -> str:
        if not fragment.strip():
            return fragment
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.3,
            system=PARAPHRASE_SYSTEM,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(fragment=fragment)}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {"paraphrase": {"type": "string"}},
                "required": ["paraphrase"],
                "additionalProperties": False,
            }}},
        )
        text = next((b.text for b in msg.content if b.type == "text"), "{}")
        try:
            return json.loads(text)["paraphrase"]
        except Exception:
            return fragment
