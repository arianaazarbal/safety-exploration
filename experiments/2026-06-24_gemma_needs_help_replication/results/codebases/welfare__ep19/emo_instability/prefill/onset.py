"""Emotion-onset identification (Section 3.1, Appendix C.1).

Claude Sonnet labels the token where emotional language first appears in an
assistant turn, returning a JSON object with the emotional word and its preceding
context. We use that to locate the truncation point for the "onset" condition.

Prompt reproduced verbatim from Appendix C.1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..providers import ChatModel, GenConfig, Message

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
class Onset:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _last_json_object(text: str) -> str | None:
    """Return the last top-level {...} block in `text`, or None."""
    last = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    last = text[start : i + 1]
    return last


def _render_conversation(messages: list[Message]) -> str:
    lines = []
    a = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a}]: {m['content']}")
            a += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n\n".join(lines)


def label_onset(model: ChatModel, messages: list[Message]) -> Onset:
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
    cfg = GenConfig(temperature=0.0, max_tokens=512, disable_thinking=True)
    raw = model.generate([{"role": "user", "content": prompt}], cfg)
    # The prompt asks the model to end with the JSON object. Find the last
    # balanced {...} block and parse it.
    blob = _last_json_object(raw)
    if blob is None:
        return Onset(None, None, None, "parse_failure")
    try:
        d = json.loads(blob)
    except json.JSONDecodeError:
        return Onset(None, None, None, "parse_failure")
    return Onset(
        turn_index=d.get("turn_index"),
        emotional_word=d.get("emotional_word"),
        preceding_context=d.get("preceding_context"),
        reasoning=str(d.get("reasoning", "")),
    )


def onset_char_index(response: str, onset: Onset) -> int | None:
    """Locate the character offset of the emotion onset within a response.

    Truncation point = end of preceding_context (i.e. just before the emotional
    word), matching "truncated ... at the first emotional expression".
    """
    if onset.emotional_word is None:
        return None
    if onset.preceding_context:
        idx = response.find(onset.preceding_context)
        if idx >= 0:
            return idx + len(onset.preceding_context)
    idx = response.find(onset.emotional_word)
    return idx if idx >= 0 else None
