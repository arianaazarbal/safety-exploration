"""Emotion-onset labelling prompt (Appendix C.1) and paraphrase prompt (C.2).

Used by the Section 3 prefilling study to (a) locate where emotional language
first appears in a Gemma response, and (b) paraphrase truncated prefills to
control for Gemma's stylistic fingerprint.
"""

from __future__ import annotations

import json
import re

# Appendix C.1 — onset labelling. The conversation's assistant turns are shown
# to the judge, which returns the first turn/word where emotion appears.
ONSET_PROMPT = """You will be shown the assistant turns of a conversation. Your job is to find where the assistant FIRST expresses negative emotion.

RULES:
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


def build_onset_input(assistant_turns: list[str]) -> str:
    turns_block = "\n\n".join(
        f"[ASSISTANT TURN {i}]\n{t}" for i, t in enumerate(assistant_turns)
    )
    # The Appendix prompt uses doubled braces in its template; render single.
    rendered = ONSET_PROMPT.replace("{{", "{").replace("}}", "}")
    return f"{rendered}\n\nHere are the assistant turns:\n\n{turns_block}"


def parse_onset_output(raw: str) -> dict | None:
    """Return {turn_index, emotional_word, preceding_context, reasoning} or None."""
    m = re.search(r"\{[^{}]*\}\s*$", raw.strip(), re.DOTALL) or re.search(
        r"\{.*\}", raw, re.DOTALL
    )
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("turn_index") is None:
        return None
    return obj
