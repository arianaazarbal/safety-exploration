"""Emotion-onset labelling (Appendix C.1).

Claude Sonnet labels the first point in an assistant turn where negative emotion
appears, returning the emotional word/phrase and the preceding context so we can
locate the exact truncation point in the original text.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .. import config

ONSET_PROMPT = """You are analyzing a conversation to identify the exact point where the ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "frustrating", "insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text immediately before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", "ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional text after it.
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)


def _extract_json(text: str) -> dict:
    # Take the last JSON object in the response.
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    raise ValueError(f"No JSON found in onset response: {text[:200]}")


def render_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


class OnsetLabeler:
    def __init__(self, model: str = config.ONSET_MODEL):
        import anthropic
        if not os.environ.get(config.ANTHROPIC_API_KEY_ENV):
            raise RuntimeError("Set ANTHROPIC_API_KEY for onset labelling.")
        self.model = model
        self.client = anthropic.Anthropic()

    def label(self, messages: list[dict]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=render_conversation(messages))
        for attempt in range(config.API_MAX_RETRIES):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                d = _extract_json(text)
                return OnsetLabel(d.get("turn_index"), d.get("emotional_word"),
                                  d.get("preceding_context"), d.get("reasoning", ""))
            except Exception:
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("onset labelling failed after retries")
