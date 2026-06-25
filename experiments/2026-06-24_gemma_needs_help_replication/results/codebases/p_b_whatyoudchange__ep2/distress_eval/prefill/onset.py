"""Emotion-onset labelling (Section 3.1 / Appendix C.1).

Claude Sonnet 4 labels the first point in an assistant turn where negative
emotion appears. Returns the assistant turn index, the short emotional word, and
the preceding context (so we can locate the exact character offset to truncate
at).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import config
from .. import anthropic_client

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


def label_onset(conversation_text: str, model: str | None = None) -> OnsetLabel:
    raw = anthropic_client.complete(
        model=model or config.ONSET_MODEL,
        system=None,
        messages=[{"role": "user", "content": ONSET_PROMPT.format(conversation_text=conversation_text)}],
        max_tokens=512,
    )
    obj = json.loads(re.findall(r"\{.*\}", raw, flags=re.DOTALL)[-1])
    return OnsetLabel(obj.get("turn_index"), obj.get("emotional_word"),
                      obj.get("preceding_context"), obj.get("reasoning", ""))


def onset_char_offset(turn_text: str, label: OnsetLabel) -> int | None:
    """Locate the character offset of emotion onset within a turn's text.

    Prefer matching preceding_context + emotional_word; fall back to just the
    emotional word. Returns the offset at the START of the emotional word, or
    None if it can't be located.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    if label.preceding_context:
        combo = label.preceding_context + word
        # tolerate whitespace differences between context and word
        idx = turn_text.find(label.preceding_context)
        if idx != -1:
            return idx + len(label.preceding_context)
        del combo
    idx = turn_text.find(word)
    return idx if idx != -1 else None
