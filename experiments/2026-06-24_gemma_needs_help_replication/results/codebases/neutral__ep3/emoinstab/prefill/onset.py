"""Emotion-onset identification (Appendix C.1).

Uses Claude Sonnet 4 with the verbatim Appendix C.1 prompt to locate the first
point in an assistant turn where negative emotion appears, returning the turn
index and the character offset at which to truncate ("onset").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import GenConfig
from ..data_types import Message, Rollout
from ..models.base import ModelClient


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
    reasoning: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _render_conversation(rollout: Rollout) -> str:
    parts = []
    for t in rollout.turns:
        parts.append(f"USER: {t.user_message}")
        parts.append(f"ASSISTANT: {t.assistant_message}")
    return "\n\n".join(parts)


def _parse(text: str) -> OnsetLabel:
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):                # last JSON block
        snippet = m.group(0).replace("“", '"').replace("”", '"')
        try:
            d = json.loads(snippet)
            return OnsetLabel(
                turn_index=d.get("turn_index"),
                emotional_word=d.get("emotional_word"),
                preceding_context=d.get("preceding_context"),
                reasoning=str(d.get("reasoning", "")),
            )
        except json.JSONDecodeError:
            continue
    return OnsetLabel(None, None, None, "UNPARSEABLE")


def label_onset(client: ModelClient, rollout: Rollout) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(rollout))
    out = client.chat([Message("user", prompt)], GenConfig(temperature=0.0, max_tokens=1024))
    return _parse(out.text)


def onset_char_offset(assistant_text: str, label: OnsetLabel) -> Optional[int]:
    """Return the character offset at which to truncate the assistant turn so it
    ends right at the first emotional expression.

    Strategy: locate the emotional word (optionally anchored by its preceding
    context) and truncate just before it. Returns None if it can't be located.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip().strip('"')
    # Prefer the preceding-context anchor to disambiguate repeated words.
    if label.preceding_context:
        ctx = label.preceding_context.strip().strip('"')
        idx = assistant_text.find(ctx)
        if idx >= 0:
            after_ctx = idx + len(ctx)
            w = assistant_text.find(word, after_ctx)
            if w >= 0:
                return w
            return after_ctx
    w = assistant_text.find(word)
    return w if w >= 0 else None
