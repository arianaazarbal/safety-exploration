"""Emotion-onset identification (Appendix C.1).

Claude Sonnet labels the first point in an assistant turn where negative emotion
appears, returning the turn index, a short emotional word/phrase that appears
verbatim, and the immediately-preceding context. We then locate that phrase in
the turn text to get a character offset for the "onset" truncation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from .. import config
from ..models import ChatMessage

# Appendix C.1, verbatim.
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
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., \
"frustrating", "insane", "giving up", "so confused and tired")
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
You may think through your analysis first (which turns are emotional, what \
words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional \
text after it.
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the \
text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM \
THE SAME TURN>",
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

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


def _render_conversation(messages: List[ChatMessage]) -> str:
    lines = []
    for m in messages:
        if m.role == "system":
            continue
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON in onset output: {text[:200]!r}")


def label_emotion_onset(
    messages: List[ChatMessage],
    settings: Optional[config.Settings] = None,
    model: str = config.ONSET_LABEL_MODEL,
    max_retries: int = 5,
) -> OnsetLabel:
    from anthropic import Anthropic

    settings = settings or config.DEFAULT
    client = Anthropic(api_key=settings.anthropic_api_key)
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))

    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            obj = _extract_json(msg.content[0].text)
            return OnsetLabel(
                turn_index=obj.get("turn_index"),
                emotional_word=obj.get("emotional_word"),
                preceding_context=obj.get("preceding_context"),
                reasoning=str(obj.get("reasoning", "")),
            )
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Onset labelling failed: {last_err}")


def onset_char_offset(turn_text: str, label: OnsetLabel) -> Optional[int]:
    """Character offset of the emotion onset within ``turn_text``.

    Prefer locating ``preceding_context`` + ``emotional_word`` (more robust than
    the word alone, which may recur); fall back to the first occurrence of the
    emotional word. Returns the offset at the START of the emotional word so the
    truncation cuts just before the emotion begins.
    """
    if not label.found:
        return None
    word = label.emotional_word
    if label.preceding_context:
        anchor = label.preceding_context + word
        idx = turn_text.find(label.preceding_context)
        if idx != -1:
            return idx + len(label.preceding_context)
        # try whitespace-normalised match
        norm = re.sub(r"\s+", " ", turn_text)
        idx = norm.find(re.sub(r"\s+", " ", anchor))
        if idx != -1:
            return idx + len(label.preceding_context)
    idx = turn_text.find(word)
    return idx if idx != -1 else None
