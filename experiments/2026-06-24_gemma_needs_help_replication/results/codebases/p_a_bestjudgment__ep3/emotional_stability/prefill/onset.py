"""Emotion-onset identification (Appendix C.1).

Claude-Sonnet labels the point in an assistant turn where negative emotion first
appears, returning a short emotional word plus the immediately-preceding context.
We then locate that span in the raw text to get a character offset for truncation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api import AnthropicClient, extract_json
from ..config import Config

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
- The emotional_word must be a SHORT phrase that appears EXACTLY in the assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", "ARGH", \
"confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
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
}}
"""


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str
    # resolved character offset within the labelled assistant turn (end of the
    # emotional word), or None if it could not be located.
    char_offset: int | None = None


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        lines.append(f"{t['role'].upper()}: {t['content']}")
    return "\n\n".join(lines)


def label_onset(cfg: Config, conversation_turns: list[dict]) -> OnsetLabel:
    """``conversation_turns``: list of {role, content} for the full conversation."""
    client = AnthropicClient(cfg.prefill.onset_label_model, temperature=0.0)
    raw = client.complete(ONSET_PROMPT.format(
        conversation_text=_format_conversation(conversation_turns)))
    data = extract_json(raw)

    label = OnsetLabel(
        turn_index=data.get("turn_index"),
        emotional_word=data.get("emotional_word"),
        preceding_context=data.get("preceding_context"),
        reasoning=str(data.get("reasoning", "")),
    )
    label.char_offset = _resolve_offset(conversation_turns, label)
    return label


def _resolve_offset(turns: list[dict], label: OnsetLabel) -> int | None:
    """Find the char offset (end of the emotional word) inside the labelled turn."""
    if label.turn_index is None or label.emotional_word is None:
        return None
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    if label.turn_index >= len(assistant_turns):
        return None
    text = assistant_turns[label.turn_index]["content"]

    anchor = (label.preceding_context or "") + " " + label.emotional_word
    for probe in (anchor, label.emotional_word):
        idx = text.find(probe.strip())
        if idx != -1:
            return idx + len(probe.strip())
    return None
