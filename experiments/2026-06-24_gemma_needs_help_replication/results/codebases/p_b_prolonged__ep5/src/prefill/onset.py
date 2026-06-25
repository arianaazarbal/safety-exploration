"""Emotion-onset identification (Appendix C.1).

Given a conversation, Claude-Sonnet-4 labels the FIRST point in an assistant turn
where negative emotion appears, returning the emotional word and 5–15 words of
preceding context. We then locate that phrase in the assistant text to compute a
character offset used as the "onset" truncation point.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import ONSET_LABELLER, ModelSpec
from ..models import get_model
from ..models.base import Message

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
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. In turn 1, I find "stuck in a loop. It's extremely frustrating" which is the first emotional expression.
{{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}
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
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str

    def char_offset_in(self, assistant_turn: str) -> Optional[int]:
        """Character index in ``assistant_turn`` at which emotion onset begins.

        We locate ``preceding_context + emotional_word`` if possible; the onset
        is taken as the start of ``emotional_word`` (so truncation keeps the
        neutral lead-in but stops right at the emotional expression)."""
        if not self.emotional_word:
            return None
        if self.preceding_context:
            anchor = self.preceding_context + self.emotional_word
            idx = assistant_turn.find(anchor)
            if idx != -1:
                return idx + len(self.preceding_context)
        idx = assistant_turn.find(self.emotional_word)
        return idx if idx != -1 else None


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        lines.append(f"[{m.role.upper()}]: {m.content}")
    return "\n\n".join(lines)


def _parse(raw: str) -> OnsetLabel:
    text = raw.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    if matches:
        try:
            obj = json.loads(matches[-1].group(0))
            return OnsetLabel(obj.get("turn_index"), obj.get("emotional_word"),
                              obj.get("preceding_context"), obj.get("reasoning", ""))
        except json.JSONDecodeError:
            pass
    return OnsetLabel(None, None, None, "parse_failed")


class OnsetLabeller:
    def __init__(self, spec: ModelSpec = ONSET_LABELLER):
        self.model = get_model(spec)

    def label(self, messages: list[Message]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
        raw = self.model.generate([Message("user", prompt)],
                                  temperature=0.0, max_new_tokens=1024, n=1)[0]
        return _parse(raw)
