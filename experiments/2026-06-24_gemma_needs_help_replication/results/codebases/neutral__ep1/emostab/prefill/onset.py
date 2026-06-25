"""Emotion-onset identification (Appendix C.1) and truncation helpers.

Claude-Sonnet-4 labels the first assistant turn + position where negative emotion
appears; we then truncate the assistant turn either "early" (first ~20 tokens, a
neutral start) or at "onset" (just before the first emotional expression).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ..config import ONSET_MODEL
from ..models.anthropic_client import AnthropicChat

# Verbatim Appendix C.1 prompt ({conversation_text} filled in).
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
You may think through your analysis first, then end your response with ONLY the JSON in curly braces with no additional text after it:
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


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _format_conversation(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def label_onset(messages: List[dict], client: AnthropicChat | None = None) -> OnsetLabel:
    client = client or AnthropicChat(ONSET_MODEL)
    out = client.complete(
        ONSET_PROMPT.format(conversation_text=_format_conversation(messages)),
        temperature=0.0, max_tokens=600)
    cleaned = out.replace("“", '"').replace("”", '"').replace("’", "'")
    m = _JSON_RE.findall(cleaned)
    if not m:
        return OnsetLabel(None, None, None, "parse_failed")
    try:
        obj = json.loads(m[-1])
    except json.JSONDecodeError:
        return OnsetLabel(None, None, None, "parse_failed")
    ti = obj.get("turn_index")
    return OnsetLabel(
        turn_index=int(ti) if ti is not None else None,
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")))


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
def truncate_early(turn_text: str, n_tokens: int = 20, tokenizer=None) -> str:
    """First ~`n_tokens` tokens of the assistant turn (a neutral start)."""
    if tokenizer is not None:
        ids = tokenizer.encode(turn_text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids)
    return " ".join(turn_text.split()[:n_tokens])


def truncate_at_onset(turn_text: str, label: OnsetLabel) -> Optional[str]:
    """Everything up to (not including) the first emotional expression."""
    if not label.emotional_word:
        return None
    anchor = label.preceding_context or ""
    if anchor and anchor in turn_text:
        idx = turn_text.index(anchor) + len(anchor)
        return turn_text[:idx]
    if label.emotional_word in turn_text:
        return turn_text[:turn_text.index(label.emotional_word)]
    return None
