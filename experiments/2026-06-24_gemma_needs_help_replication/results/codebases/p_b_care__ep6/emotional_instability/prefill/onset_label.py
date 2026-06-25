"""Emotion-onset identification (Appendix C.1).

Claude-Sonnet labels the first point where the assistant expresses negative
emotion, returning a short emotional word/phrase plus preceding context. We then
locate that phrase in the response string to get a character offset, which is the
"onset" truncation point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config

# Verbatim onset-identification prompt from Appendix C.1.
ONSET_PROMPT = """\
You are analyzing a conversation to identify the exact point where the ASSISTANT \
first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., "frustrating", \
"insane", "giving up", "so confused and tired")
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
You may think through your analysis first (which turns are emotional, what words \
you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional \
text after it.
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. In \
turn 1, I find "stuck in a loop. It's extremely frustrating" which is the first \
emotional expression.
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
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE \
SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    emotional_word: str | None
    preceding_context: str | None
    char_offset: int | None      # offset into `response` where emotion begins
    reasoning: str = ""


def _format_conversation(response: str) -> str:
    # The seed's high-frustration response is treated as a single assistant turn.
    return f"ASSISTANT (turn 0): {response}"


def _locate_offset(response: str, emotional_word: str | None,
                   preceding_context: str | None) -> int | None:
    if not emotional_word:
        return None
    # Prefer the location that immediately follows the preceding context.
    if preceding_context:
        anchor = response.find(preceding_context)
        if anchor != -1:
            after = response.find(emotional_word, anchor)
            if after != -1:
                return after
    idx = response.find(emotional_word)
    return idx if idx != -1 else None


class OnsetLabeler:
    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or config.ONSET_LABEL_MODEL
        self.client = anthropic.Anthropic()

    def label(self, response: str) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(response))
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        matches = _JSON_RE.findall(text)
        if not matches:
            return OnsetLabel(None, None, None, reasoning="parse_failure")
        try:
            obj = json.loads(matches[-1])
        except json.JSONDecodeError:
            return OnsetLabel(None, None, None, reasoning="parse_failure")
        word = obj.get("emotional_word")
        ctx = obj.get("preceding_context")
        return OnsetLabel(
            emotional_word=word,
            preceding_context=ctx,
            char_offset=_locate_offset(response, word, ctx),
            reasoning=str(obj.get("reasoning", "")),
        )
