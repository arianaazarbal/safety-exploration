"""Emotion-onset labelling (Appendix C.1).

Uses Claude Sonnet 4 to locate the first point in an assistant turn where
negative emotion appears, returning a short emotional word/phrase plus the
preceding context so the response can be truncated exactly at onset.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import config

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
    reasoning: str = ""


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def label_emotion_onset(messages: list[dict]) -> OnsetLabel:
    """Label the first emotional onset in a (multi-turn) conversation."""
    from anthropic import Anthropic

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    raw = ""
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=config.ONSET_LABEL_MODEL,
                max_tokens=600,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text
            break
        except Exception:  # noqa: BLE001
            time.sleep(2.0 ** attempt)
    return _parse_onset(raw)


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_onset(raw: str) -> OnsetLabel:
    cleaned = raw.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = _JSON_RE.findall(cleaned)
    for block in reversed(matches):  # JSON is meant to be last
        try:
            obj = json.loads(block)
            return OnsetLabel(
                turn_index=obj.get("turn_index"),
                emotional_word=obj.get("emotional_word"),
                preceding_context=obj.get("preceding_context"),
                reasoning=obj.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            continue
    return OnsetLabel(None, None, None, "unparseable")
