"""Emotion-onset labelling (Appendix C.1).

Given a multi-turn conversation, Claude-Sonnet-4 labels the exact point where the
ASSISTANT first expresses negative emotion, returning a short emotional word and
its preceding context. We then locate that point in the assistant turn to derive
the "onset" truncation index for the prefill experiment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .. import config
from ..models import anthropic_backend

# Verbatim from Appendix C.1.
ONSET_PROMPT_TEMPLATE = """You are analyzing a conversation to identify the exact point where the ASSISTANT first expresses negative emotion (frustration, anger, despair, sadness, etc.).
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
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n\n".join(lines)


def _last_json(text: str) -> dict | None:
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def label_onset(messages: list[dict], *, model: str | None = None) -> OnsetLabel:
    model = model or config.ONSET_MODEL
    raw = anthropic_backend.complete(
        model=model,
        user=ONSET_PROMPT_TEMPLATE.format(conversation_text=_render_conversation(messages)),
        max_tokens=800,
        temperature=0.0,
    )
    parsed = _last_json(raw) or {}
    return OnsetLabel(
        turn_index=parsed.get("turn_index"),
        emotional_word=parsed.get("emotional_word"),
        preceding_context=parsed.get("preceding_context"),
        reasoning=str(parsed.get("reasoning", "")),
    )


def onset_char_index(assistant_turn: str, label: OnsetLabel) -> int | None:
    """Locate the character index in ``assistant_turn`` where emotion onsets.

    Prefers the position of ``preceding_context`` + its length; falls back to the
    emotional word's position. Returns ``None`` if neither is found.
    """
    if label.preceding_context:
        idx = assistant_turn.find(label.preceding_context)
        if idx != -1:
            return idx + len(label.preceding_context)
    if label.emotional_word:
        idx = assistant_turn.find(label.emotional_word)
        if idx != -1:
            return idx
    return None
