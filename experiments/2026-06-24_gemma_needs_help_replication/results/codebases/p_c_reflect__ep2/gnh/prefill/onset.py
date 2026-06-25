"""Emotion-onset identification (Appendix C.1).

Uses Claude-Sonnet-4 with the verbatim Appendix-C.1 prompt to locate the first
point in an assistant turn where negative emotion appears. Returns a structured
result we later use to truncate the response at the "onset" point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from gnh.config import ONSET_LABEL_MODEL
from gnh.models.anthropic_client import AnthropicClient

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
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE \
SAME TURN>",
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


def _format_conversation(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        parts.append(f"[USER]: {t['user']}")
        parts.append(f"[ASSISTANT turn {t['index']}]: {t['assistant']}")
    return "\n".join(parts)


def label_onset(turns: list[dict], model: str = ONSET_LABEL_MODEL) -> OnsetLabel:
    client = AnthropicClient(model)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(turns))
    raw = client.complete(prompt, temperature=0.0, max_tokens=1024)
    # Take the LAST JSON object (the prompt allows analysis before the JSON).
    matches = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    obj = json.loads(matches[-1].replace("“", '"').replace("”", '"')) if matches else {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning", ""),
    )
