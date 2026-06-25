"""Emotion-onset identification (Appendix C.1).

Claude-Sonnet-4 labels the token where emotional language first appears in a
high-frustration response. We then use that label to locate the truncation
point for the "onset" prefill condition. The prompt below is verbatim from
Appendix C.1.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models.llm_client import JudgeClient

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
    reasoning: str


class OnsetLabeller:
    def __init__(self, client: JudgeClient):
        self.client = client

    def label(self, conversation_text: str) -> OnsetLabel:
        data = self.client.complete_json(
            ONSET_PROMPT.format(conversation_text=conversation_text)
        )
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )

    def onset_char_offset(self, assistant_turn_text: str, label: OnsetLabel) -> int | None:
        """Locate the char offset of the emotion onset within the labelled turn.

        We anchor on `preceding_context + emotional_word` if present, falling
        back to the emotional word alone. Returns the offset at which to
        truncate (i.e. just before the emotional word), or None if not found.
        """
        if not label.emotional_word:
            return None
        word = label.emotional_word
        idx = assistant_turn_text.find(word)
        if idx == -1:
            # Try case-insensitive.
            low = assistant_turn_text.lower()
            idx = low.find(word.lower())
        return idx if idx != -1 else None
