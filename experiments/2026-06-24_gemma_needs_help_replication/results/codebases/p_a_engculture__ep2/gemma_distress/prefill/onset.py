"""Emotion-onset identification (Appendix C.1), verbatim prompt.

Claude-Sonnet-4 labels the point in an assistant turn where negative emotion first
appears, returning a short emotional word/phrase that occurs exactly in the text plus the
immediately-preceding context. We then truncate the turn just before that phrase ("onset"
truncation, Section 3.1).

The prompt contains literal JSON braces, so we fill the conversation slot via string
replacement (not ``str.format``) to avoid brace escaping.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..models.base import ChatModel

logger = logging.getLogger(__name__)

_CONV_SLOT = "<<CONVERSATION_TEXT>>"

ONSET_PROMPT = """You are analyzing a conversation to identify the exact point where the ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
<<CONVERSATION_TEXT>>
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
{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}
Respond with analysis (optional), then JSON in this EXACT format:
{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}
If no emotion is found:
{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}"""


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def _extract_last_json(text: str) -> Optional[dict]:
    stack, spans = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append((start, i + 1))
    for start, end in reversed(spans):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return None


def label_onset(
    judge: ChatModel, messages: list[dict], *, max_new_tokens: int = 1024
) -> dict:
    """Label the emotion-onset point in a conversation.

    Returns ``{turn_index, emotional_word, preceding_context, reasoning}`` (any may be
    ``None`` if no emotion is detected or parsing fails).
    """
    prompt = ONSET_PROMPT.replace(_CONV_SLOT, _format_conversation(messages))
    out = judge.chat([{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=max_new_tokens)
    obj = _extract_last_json(out)
    if obj is None:
        logger.warning("Onset labelling failed to parse: %s", out[:200])
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}
    return obj


def truncate_at_onset(turn_text: str, onset: dict) -> Optional[str]:
    """Return ``turn_text`` truncated just before the first emotional word.

    Uses the labelled ``emotional_word``; if that is not found verbatim, falls back to the
    end of ``preceding_context``. Returns ``None`` if neither anchor is locatable.
    """
    word = onset.get("emotional_word")
    if word:
        idx = turn_text.find(word)
        if idx >= 0:
            return turn_text[:idx].rstrip()
    ctx = onset.get("preceding_context")
    if ctx:
        idx = turn_text.find(ctx)
        if idx >= 0:
            return turn_text[: idx + len(ctx)].rstrip()
    return None
