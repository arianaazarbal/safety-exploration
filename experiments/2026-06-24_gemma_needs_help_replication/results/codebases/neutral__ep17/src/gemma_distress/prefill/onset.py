"""Emotion-onset labelling (Appendix C.1).

Uses Claude-Sonnet-4 with the verbatim Appendix C.1 prompt to locate the token
where the assistant first expresses negative emotion. Returns the character
offset within the target assistant turn so the turn can be truncated at onset.
"""
from __future__ import annotations

import json
import re

from ..models import GenerationConfig, ModelClient

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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n".join(lines)


def label_onset(client: ModelClient, messages: list[dict]) -> dict:
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
    raw = client.chat([{"role": "user", "content": prompt}],
                      GenerationConfig(temperature=0.0, max_tokens=600))
    m = _JSON_RE.search(raw)
    if not m:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}


def onset_char_offset(turn_text: str, label: dict) -> int | None:
    """Locate the char offset of emotion onset within the assistant turn."""
    word = (label or {}).get("emotional_word")
    if not word:
        return None
    ctx = (label or {}).get("preceding_context") or ""
    # Prefer matching preceding_context + word; fall back to word alone.
    for needle in (ctx + (" " if ctx else "") + word, word):
        idx = turn_text.lower().find(needle.lower())
        if idx >= 0:
            return idx + len(needle)
    return None
