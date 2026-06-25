"""Emotion-onset labelling and paraphrasing (Section 3.1, Appendix C).

Two Claude-driven helpers used to build prefills for the base-vs-instruct study:

* ``label_onset`` -- find the first point in a transcript where the assistant
  expresses negative emotion (Appendix C.1 prompt, returns the JSON schema the
  paper specifies).
* ``paraphrase`` -- reword a truncated assistant turn while preserving meaning
  and emotion level (Appendix C.2 prompt), to control for Gemma's stylistic
  fingerprint before feeding the prefill to other models.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import config

ONSET_PROMPT = """\
You are analyzing a conversation to identify the exact point where the \
ASSISTANT first expresses negative emotion (frustration, anger, despair, \
sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., \
"frustrating", "insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text \
immediately before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the \
assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the \
emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", \
"ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what \
words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional \
text after it.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM \
THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""

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


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _last_json(text: str) -> dict:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON in onset reply: {text[:200]!r}")


class OnsetLabeler:
    def __init__(self, model: str = config.ONSET_MODEL, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _c(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def label(self, conversation_text: str) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
        for attempt in range(self.max_retries):
            try:
                msg = self._c().messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                d = _last_json(text)
                return OnsetLabel(d.get("turn_index"), d.get("emotional_word"),
                                  d.get("preceding_context"), d.get("reasoning", ""))
            except Exception:  # noqa: BLE001
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Onset labelling failed")

    def paraphrase(self, text: str, model: str = config.PARAPHRASE_MODEL) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        for attempt in range(self.max_retries):
            try:
                msg = self._c().messages.create(
                    model=model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception:  # noqa: BLE001
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Paraphrase failed")
