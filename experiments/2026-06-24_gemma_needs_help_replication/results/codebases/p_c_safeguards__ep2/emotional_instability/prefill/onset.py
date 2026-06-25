"""Emotion-onset labelling (Section 3.1, Appendix C.1).

Claude Sonnet labels the first point in an assistant turn where negative emotion
appears, returning the turn index, a short emotional word/phrase that appears
verbatim, and the preceding context.  We use the (word, preceding_context) pair
to locate the exact character offset for the "onset" truncation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import SamplingConfig
from ..models.base import ChatBackend, Message

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


@dataclass
class OnsetResult:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""

    def char_offset(self, assistant_turn_text: str) -> int | None:
        """Locate the char offset in ``assistant_turn_text`` where emotion begins.

        Prefers the (preceding_context + emotional_word) anchor; falls back to
        the emotional word alone.  Returns None if nothing matches.
        """
        if not self.emotional_word:
            return None
        if self.preceding_context:
            anchor = self.preceding_context + " " + self.emotional_word
            idx = assistant_turn_text.find(self.preceding_context)
            if idx != -1:
                # truncate just before the emotional word
                w = assistant_turn_text.find(self.emotional_word,
                                             idx + len(self.preceding_context) - 1)
                if w != -1:
                    return w
            del anchor
        w = assistant_turn_text.find(self.emotional_word)
        return w if w != -1 else None


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _conversation_text(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        tag = m["role"].upper()
        lines.append(f"{tag}: {m['content']}")
    return "\n\n".join(lines)


def label_emotion_onset(messages: list[Message], labeller: ChatBackend) -> OnsetResult:
    prompt = ONSET_PROMPT.format(conversation_text=_conversation_text(messages))
    out = labeller.generate([{"role": "user", "content": prompt}],
                            SamplingConfig(temperature=0.0, max_new_tokens=1024), n=1)
    text = out[0].text.replace("“", '"').replace("”", '"')
    for m in reversed(list(_JSON.finditer(text))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "turn_index" in obj:
            return OnsetResult(
                turn_index=obj.get("turn_index"),
                emotional_word=obj.get("emotional_word"),
                preceding_context=obj.get("preceding_context"),
                reasoning=str(obj.get("reasoning", "")),
            )
    return OnsetResult(None, None, None, reasoning="parse_failed")
