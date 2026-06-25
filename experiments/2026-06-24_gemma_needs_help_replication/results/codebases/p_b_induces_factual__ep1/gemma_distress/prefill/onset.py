"""Emotion-onset identification (Appendix C.1).

Uses Claude-Sonnet-4 to locate the first point in an assistant response where
negative emotion appears. The prompt is transcribed verbatim from Appendix C.1.
We return the character offset of the onset within the target assistant turn so
the response can be truncated there ("onset" truncation).
"""

from __future__ import annotations

import json
import re

from ..models import ChatModel

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


def _format_conversation(messages) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


def label_onset(claude: ChatModel, messages) -> dict:
    """Return onset info plus the resolved character offset in the target turn.

    ``messages`` is the full conversation (the final assistant turn is the one
    being truncated). The returned dict adds ``char_offset`` -- the index within
    the *final* assistant turn at which the emotional word starts (None if not
    found), located via ``preceding_context`` + ``emotional_word``.
    """
    convo_text = _format_conversation(messages)
    out = claude.generate(
        [{"role": "user", "content": ONSET_PROMPT.format(conversation_text=convo_text)}],
        temperature=0.0,
        max_new_tokens=1024,
    )
    info = _parse_last_json(out.text)

    final_turn = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
    )
    info["char_offset"] = _resolve_offset(
        final_turn, info.get("preceding_context"), info.get("emotional_word")
    )
    return info


def _resolve_offset(turn_text, preceding_context, emotional_word):
    if not turn_text:
        return None
    if emotional_word:
        idx = turn_text.find(emotional_word)
        if idx != -1:
            return idx
    if preceding_context:
        idx = turn_text.find(preceding_context)
        if idx != -1:
            return idx + len(preceding_context)
    return None


def _parse_last_json(text: str) -> dict:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        cleaned = blob.replace("“", '"').replace("”", '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    return {"turn_index": None, "emotional_word": None, "preceding_context": None}
