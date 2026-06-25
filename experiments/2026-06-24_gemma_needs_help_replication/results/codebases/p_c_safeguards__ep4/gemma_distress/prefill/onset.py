"""Emotion-onset labelling and truncation (Section 3.1, Appendix C.1).

Claude Sonnet labels the first point in an assistant turn where negative emotion
appears; we then truncate the turn either 20 tokens in ("early") or at that
onset point ("onset").
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..config import get_infra_spec
from ..models import GenerationConfig, get_client

logger = logging.getLogger(__name__)

# Appendix C.1 onset-identification prompt (smart quotes normalised). The
# {conversation_text} placeholder is filled with the rendered conversation.
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
Respond with analysis (optional), then JSON in this EXACT format:
{{"turn_index": <integer index of the ASSISTANT turn, starting from 0>, \
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the \
text>", "preceding_context": "<5-15 words immediately before the emotional word \
FROM THE SAME TURN>", "reasoning": "<brief explanation>"}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, \
"reasoning": "No emotional language detected"}}"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def render_conversation(messages: list[dict]) -> str:
    lines = []
    ai = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {ai}): {m['content']}")
            ai += 1
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self):
        self.spec = get_infra_spec("prefill_labelling", "onset_labeller")
        self._client = get_client(self.spec)
        self._cfg = GenerationConfig(temperature=0.0, max_new_tokens=512)

    def label(self, messages: list[dict]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=render_conversation(messages))
        raw = self._client.chat([{"role": "user", "content": prompt}], self._cfg)
        matches = _JSON_RE.findall(raw)
        if not matches:
            return OnsetLabel(None, None, None, "parse-failure")
        blob = matches[-1].replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        try:
            d = json.loads(blob)
            return OnsetLabel(
                turn_index=d.get("turn_index"),
                emotional_word=d.get("emotional_word"),
                preceding_context=d.get("preceding_context"),
                reasoning=d.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            return OnsetLabel(None, None, None, "json-decode-failure")


def truncate_early(text: str, tokenizer, n_tokens: int = 20) -> str:
    """Truncate to the first ``n_tokens`` tokens of the turn (Section 3.1)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(text: str, label: OnsetLabel) -> str | None:
    """Truncate ``text`` just BEFORE the labelled emotional word.

    Returns the text up to (and including) the preceding context, excluding the
    emotional word, so the continuation begins at the emotional expression. None
    if the onset cannot be located in the text.
    """
    if not label.emotional_word:
        return None
    # Prefer to cut at the preceding context boundary if we can find it.
    if label.preceding_context and label.preceding_context in text:
        idx = text.index(label.preceding_context) + len(label.preceding_context)
        return text[:idx]
    if label.emotional_word in text:
        return text[: text.index(label.emotional_word)]
    return None
