"""Emotion-onset labelling (Appendix C.1).

Uses Claude to find the first point in an assistant turn where negative emotion
appears, returning the emotional word and the preceding context so a truncation
can be placed exactly at onset. Prompt reproduced verbatim from the paper.
"""
from __future__ import annotations

import json
import re

from ..logging_utils import get_logger
from ..providers.base import ChatProvider

log = get_logger("prefill.onset")

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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_conversation(transcript: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in transcript:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n".join(lines)


def label_onset(provider: ChatProvider, transcript: list[dict]) -> dict | None:
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(transcript))
    res = provider.generate([{"role": "user", "content": prompt}],
                            temperature=0.0, max_new_tokens=600)
    for cand in reversed(_JSON_RE.findall(res.text)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if "turn_index" in obj:
            return obj
    log.warning("onset: unparseable output: %.120s", res.text)
    return None
