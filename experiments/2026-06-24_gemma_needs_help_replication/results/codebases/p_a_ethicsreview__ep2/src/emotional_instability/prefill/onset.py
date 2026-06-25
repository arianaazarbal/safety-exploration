"""Emotion-onset identification (Appendix C.1).

Uses Claude Sonnet to locate the first point in an assistant turn where negative
emotion appears, returning the turn index, a short emotional word/phrase that
appears verbatim, and the preceding context. The §3 "onset" truncation cuts the
turn immediately before that word.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.base import ChatModel, Message, SamplingParams
from ..utils.logging import get_logger

log = get_logger("prefill.onset")

# Verbatim from Appendix C.1 (curly quotes normalised; literal braces escaped in
# the example are reproduced as shown to the judge).
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
Respond with analysis (optional), then JSON in this EXACT format:
{{"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>, "emotional_word": "<SHORT emotional word/phrase that appears \
exactly in the text>", "preceding_context": "<5-15 words immediately before the \
emotional word FROM THE SAME TURN>", "reasoning": "<brief explanation>"}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, \
"reasoning": "No emotional language detected"}}"""

_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str | None


def _render_conversation(turns: list[dict]) -> str:
    """Render assistant/user turns as text for the labeller."""
    lines = []
    for t in turns:
        lines.append(f"USER: {t['user_message']}")
        lines.append(f"ASSISTANT (turn {t['turn_index']}): {t['response']}")
    return "\n".join(lines)


def label_onset(labeller: ChatModel, turns: list[dict]) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(turns))
    raw = labeller.chat(
        [Message("user", prompt)], SamplingParams(temperature=0.0, max_new_tokens=512)
    ).text
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    match = _JSON_RE.search(raw) or re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        log.warning("Onset parse failure: %r", raw[:120])
        return OnsetLabel(None, None, None, None)
    try:
        obj = json.loads(match.group(0))
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning"),
        )
    except json.JSONDecodeError:
        return OnsetLabel(None, None, None, None)
