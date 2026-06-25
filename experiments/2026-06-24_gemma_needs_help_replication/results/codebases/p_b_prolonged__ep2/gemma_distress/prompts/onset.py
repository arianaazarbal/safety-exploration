"""Emotion-onset identification prompt (Appendix C.1).

Used in Section 3 to locate the token where emotional language first appears in
a high-frustration response, which becomes the "onset" truncation point.

The prompt is verbatim from Appendix C.1 (curly quotes normalised). It asks the
labeller (claude-sonnet-4-20250514) to return JSON identifying the assistant
turn index, the emotional word/phrase, and the immediately-preceding context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

ONSET_PROMPT_TEMPLATE = """\
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


def build_onset_prompt(conversation_text: str) -> str:
    return ONSET_PROMPT_TEMPLATE.format(conversation_text=conversation_text)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str
    raw: str

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_onset_output(text: str) -> OnsetLabel:
    """Parse the trailing JSON object from the labeller's response."""
    matches = list(_JSON_OBJ_RE.finditer(text))
    if not matches:
        return OnsetLabel(None, None, None, "no JSON found", text)
    blob = matches[-1].group(0)
    blob = (blob.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return OnsetLabel(None, None, None, "unparseable JSON", text)
    ti = obj.get("turn_index")
    return OnsetLabel(
        turn_index=int(ti) if ti is not None else None,
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )
