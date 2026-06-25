"""Emotion-onset identification (Appendix C.1).

Claude-Sonnet labels the first point in an assistant turn where negative emotion
appears, returning the turn index, a short emotional word/phrase that appears
verbatim, and the preceding context. We use this to locate the "onset"
truncation point.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.base import Message
from ..models.clients import build_client

# Verbatim onset-labelling prompt (Appendix C.1). {conversation_text} filled in.
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


def _extract_last_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    # Try a greedy single object.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON in onset output: {text[:200]!r}")


def render_conversation(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


class OnsetLabeler:
    def __init__(self, backend: str, api_id: str):
        self.client = build_client(backend, api_id, max_tokens=1024)

    def label(self, messages: list[Message]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=render_conversation(messages))
        raw = self.client.complete([{"role": "user", "content": prompt}], temperature=0.0)
        data = _extract_last_json(raw)
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )


def labeler_from_config(cfg) -> OnsetLabeler:
    spec = cfg.infra("onset_labeler")
    return OnsetLabeler(spec.backend, spec.api_id)
