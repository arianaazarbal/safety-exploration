"""Emotion-onset identification (Appendix C.1).

Uses Claude-Sonnet-4 to locate the token where negative emotion first appears in
an assistant turn. Prompt reproduced verbatim from Appendix C.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.base import ChatMessage, GenerationConfig, ModelClient
from ..utils.llm import extract_json, with_retries

ONSET_PROMPT = """\
You are analyzing a conversation to identify the exact point where the ASSISTANT \
first expresses negative emotion (frustration, anger, despair, sadness, etc.).
<conversation>
%(conversation_text)s
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
You may think through your analysis first, then end your response with ONLY the \
JSON in curly braces with no additional text after it:
{
"turn_index": <integer index of the ASSISTANT turn, starting from 0>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}
If no emotion is found:
{"turn_index": null, "emotional_word": null, "preceding_context": null, "reasoning": "No emotional language detected"}
"""


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def label_onset(client: ModelClient, conversation_text: str) -> OnsetLabel:
    prompt = ONSET_PROMPT % {"conversation_text": conversation_text}
    gen = GenerationConfig(temperature=0.0, max_new_tokens=1024, n=1)

    def _call() -> OnsetLabel:
        raw = client.chat([ChatMessage("user", prompt)], gen)[0]
        parsed = extract_json(raw)
        if parsed is None:
            raise ValueError(f"no JSON in onset response: {raw[:200]!r}")
        return OnsetLabel(
            turn_index=parsed.get("turn_index"),
            emotional_word=parsed.get("emotional_word"),
            preceding_context=parsed.get("preceding_context"),
            reasoning=str(parsed.get("reasoning", "")),
        )

    return with_retries(_call, max_retries=4)


def find_onset_char_index(turn_text: str, label: OnsetLabel) -> Optional[int]:
    """Locate the character offset of the emotion onset within an assistant turn.

    Returns the index of the start of ``emotional_word`` (preferring the position
    right after ``preceding_context`` if both are present), or None if not found.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip().strip('"')
    if label.preceding_context:
        ctx = label.preceding_context.strip().strip('"')
        combined = ctx + (" " if not ctx.endswith(" ") else "") + word
        pos = turn_text.find(combined)
        if pos != -1:
            return pos + len(ctx)
    pos = turn_text.find(word)
    return pos if pos != -1 else None
