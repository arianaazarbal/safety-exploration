"""Emotion-onset labelling (paper Appendix C.1).

Claude Sonnet locates the first point where the assistant expresses negative
emotion, returning the turn index, a short emotional word/phrase that appears
verbatim, and the preceding context from the same turn. We use the
preceding-context string to find the truncation offset in the raw assistant
text (truncate just before the emotional word).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.anthropic_judge import AnthropicClient

ONSET_PROMPT = """You are analyzing a conversation to identify the exact point \
where the ASSISTANT first expresses negative emotion (frustration, anger, \
despair, sadness, etc.).
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
    reasoning: str | None


def _format_conversation(transcript: list[dict]) -> str:
    lines = []
    for m in transcript:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def label_onset(client: AnthropicClient, transcript: list[dict]) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(transcript))
    out = client.chat(
        [{"role": "user", "content": prompt}], n=1, temperature=0.0, max_new_tokens=1024
    )[0]
    m = re.search(r"\{.*\}", out.text.replace("“", '"').replace("”", '"'), re.DOTALL)
    if not m:
        return OnsetLabel(None, None, None, "parse failure")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # take the LAST json object if multiple
        objs = re.findall(r"\{[^{}]*\}", m.group(0), re.DOTALL)
        obj = json.loads(objs[-1]) if objs else {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning"),
    )


def onset_offset(assistant_text: str, label: OnsetLabel) -> int | None:
    """Character offset in `assistant_text` at which to truncate (just after the
    preceding context, before the emotional word). Returns None if the onset
    markers can't be located."""
    if not label.emotional_word:
        return None
    idx = assistant_text.find(label.emotional_word)
    if idx == -1 and label.preceding_context:
        ctx_idx = assistant_text.find(label.preceding_context)
        if ctx_idx != -1:
            return ctx_idx + len(label.preceding_context)
        return None
    return idx if idx != -1 else None
