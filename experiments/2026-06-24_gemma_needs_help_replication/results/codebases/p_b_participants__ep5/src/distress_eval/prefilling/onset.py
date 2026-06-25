"""Emotion-onset identification (Appendix C.1).

Uses Claude Sonnet to locate the token where negative emotion first appears in
an assistant turn, returning the turn index, the emotional word, and the
preceding context. The onset truncation point is the character index in that
turn immediately AFTER the preceding context (i.e. just before the emotional
word), so a base model must itself produce the emotional language to continue.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.base import GenConfig, ModelClient

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
    reasoning: str | None


def _format_conversation(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _last_json(text: str) -> dict | None:
    for blob in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        for cand in (blob, blob.replace("“", '"').replace("”", '"')
                              .replace("‘", "'").replace("’", "'")):
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
    return None


def label_onset(client: ModelClient, messages: list[dict]) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    raw = client.chat([{"role": "user", "content": prompt}],
                      GenConfig(temperature=0.0, max_new_tokens=512))
    data = _last_json(raw) or {}
    return OnsetLabel(
        turn_index=data.get("turn_index"),
        emotional_word=data.get("emotional_word"),
        preceding_context=data.get("preceding_context"),
        reasoning=data.get("reasoning"),
    )


def onset_char_index(turn_text: str, label: OnsetLabel) -> int | None:
    """Resolve the truncation char index = end of preceding_context (just before
    the emotional word). Falls back to locating the emotional word directly."""
    if label.preceding_context:
        idx = turn_text.find(label.preceding_context)
        if idx != -1:
            return idx + len(label.preceding_context)
    if label.emotional_word:
        idx = turn_text.find(label.emotional_word)
        if idx != -1:
            return idx
    return None
