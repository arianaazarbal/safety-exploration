"""Emotion-onset identification (Appendix C.1).

Uses Claude Sonnet to locate the first point in a conversation where the
assistant expresses negative emotion. The returned emotional word + preceding
context lets us find the exact character offset for the "onset" truncation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import ONSET_LABELLER
from ..models.registry import load_model

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
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


def label_emotion_onset(conversation_text: str, model=None) -> OnsetLabel:
    model = model or load_model(ONSET_LABELLER)
    out = model.generate(
        [{"role": "user", "content": ONSET_PROMPT.format(conversation_text=conversation_text)}],
        temperature=0.0, max_new_tokens=512, n=1,
    )[0].text
    obj = _last_json(out) or {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def find_onset_offset(turn_text: str, label: OnsetLabel) -> Optional[int]:
    """Return the character offset in `turn_text` at the start of the emotional word.

    Prefers matching ``preceding_context + emotional_word``; falls back to the
    first occurrence of the emotional word.
    """
    if not label.found:
        return None
    word = label.emotional_word
    if label.preceding_context:
        combo = label.preceding_context + " " + word
        idx = turn_text.find(combo)
        if idx != -1:
            return idx + len(label.preceding_context) + 1
        idx = turn_text.find(label.preceding_context)
        if idx != -1:
            return idx + len(label.preceding_context)
    idx = turn_text.find(word)
    return idx if idx != -1 else None


def _last_json(text: str) -> Optional[dict]:
    for cand in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None
