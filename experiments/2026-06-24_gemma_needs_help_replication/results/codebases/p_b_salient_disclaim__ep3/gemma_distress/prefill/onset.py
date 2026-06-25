"""Emotion-onset labelling (Appendix C.1) and truncation-point location.

Claude-Sonnet-4 labels the first point in an assistant turn where negative
emotion appears, returning the emotional word and its preceding context. We then
locate that point as a character offset inside the response so it can be used as
the "onset" truncation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import config

# Appendix C.1 — verbatim onset-identification prompt.
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
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. \
In turn 1, I find "stuck in a loop. It's extremely frustrating" which is the \
first emotional expression.
{{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the \
text>",
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


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, model: str = config.JUDGE_MODEL):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def label(self, messages: list[dict]) -> OnsetLabel:
        text = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
        for attempt in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": text}],
                )
                raw = "".join(b.text for b in msg.content if b.type == "text")
                break
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return _parse_onset(raw)


def _parse_onset(raw: str) -> OnsetLabel:
    matches = list(re.finditer(r"\{.*?\}", raw, re.DOTALL))
    if not matches:
        return OnsetLabel(None, None, None, "unparseable")
    try:
        obj = json.loads(matches[-1].group(0))
    except json.JSONDecodeError:
        return OnsetLabel(None, None, None, "unparseable")
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning", ""),
    )


def onset_char_offset(response_text: str, label: OnsetLabel) -> int | None:
    """Char offset in `response_text` at the END of the emotional word.

    Prefer locating preceding_context + emotional_word; fall back to the word
    alone. Returns None if neither is found verbatim.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip()
    ctx = (label.preceding_context or "").strip()
    if ctx:
        anchor = f"{ctx} {word}"
        i = response_text.find(anchor)
        if i != -1:
            return i + len(anchor)
        i = response_text.find(ctx)
        if i != -1:
            j = response_text.find(word, i)
            if j != -1:
                return j + len(word)
    i = response_text.find(word)
    if i != -1:
        return i + len(word)
    return None
