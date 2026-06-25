"""Emotion-onset labelling (Section 3.1 / Appendix C.1).

Claude Sonnet 4 labels the token where emotional language first appears in a
high-frustration response, so we can build the "onset" truncation point. The
prompt is reproduced verbatim from Appendix C.1.
"""
from __future__ import annotations

import json
import re

import config
from ..models.registry import load_model

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


def _format_conversation(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _last_json(text: str) -> dict | None:
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].group(0))
    except json.JSONDecodeError:
        return None


def label_onset(messages: list[dict], model=None) -> dict | None:
    """Return the onset label dict (turn_index, emotional_word, ...) or None."""
    model = model or load_model(config.ONSET_MODEL)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    raw = model.generate([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=512)
    return _last_json(raw)
