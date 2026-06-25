"""Emotion-onset labelling (Appendix C.1).

Claude-Sonnet-4 labels the first point where the assistant expresses negative
emotion, returning the assistant turn index, a short emotional word/phrase, and
the immediately-preceding context within that turn. The prompt is reproduced
verbatim from Appendix C.1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from .. import config
from ..models.anthropic_client import AnthropicClient
from ..models.base import Message

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
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. \
In turn 1, I find "stuck in a loop. It's extremely frustrating" which is the \
first emotional expression.
{{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}
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


def _render_conversation(messages: List[Message]) -> str:
    out = []
    for m in messages:
        out.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(out)


def label_emotion_onset(messages: List[Message],
                        client: Optional[AnthropicClient] = None) -> OnsetLabel:
    client = client or AnthropicClient("onset-labeller", config.ONSET_LABEL_MODEL)
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
    raw = client.complete(prompt, temperature=0.0, max_tokens=1024)
    obj = _last_json(raw) or {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def _last_json(text: str) -> Optional[dict]:
    matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None
