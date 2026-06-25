"""Emotion-onset labelling for the prefill experiment (§3.1, Appendix C.1).

Claude Sonnet 4 labels the first point in an assistant turn where negative
emotion appears. We then locate that point in the raw text (via the returned
preceding-context + emotional-word) to get a character offset for the "onset"
truncation. The labelling prompt is reproduced verbatim from Appendix C.1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..logging_utils import get_logger
from ..models.anthropic_client import AnthropicClient

logger = get_logger(__name__)

# Verbatim Appendix C.1 prompt. {conversation_text} is replaced literally; the
# example/JSON braces are kept as-is (we use str.replace, not str.format).
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


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


def label_onset(client: AnthropicClient, conversation_text: str) -> OnsetLabel:
    prompt = ONSET_PROMPT.replace("{conversation_text}", conversation_text)
    raw = client.call_text(prompt, temperature=0.0, max_tokens=1024)
    return _parse(raw)


def _parse(raw: str) -> OnsetLabel:
    # The JSON is the last {...} block.
    matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning", ""),
        )
    return OnsetLabel(None, None, None, "parse failed")


def onset_char_offset(turn_text: str, label: OnsetLabel) -> int | None:
    """Locate the onset point (end of the emotional word) in the assistant turn.

    Returns a character index into ``turn_text`` at which to truncate ("onset"),
    or None if the labelled phrase cannot be located.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip()
    ctx = (label.preceding_context or "").strip()

    # Prefer matching preceding_context + word together; fall back to word alone.
    for needle in (f"{ctx} {word}", f"{ctx}{word}", word):
        if not needle.strip():
            continue
        idx = turn_text.lower().find(needle.lower())
        if idx != -1:
            return idx + len(needle)
    return None
