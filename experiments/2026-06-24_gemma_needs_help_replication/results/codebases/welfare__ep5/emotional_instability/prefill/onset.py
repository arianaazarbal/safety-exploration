"""Emotion-onset labelling and truncation (Section 3.1, Appendix C.1).

Each high-frustration response is truncated in two places:
* **early**: 20 tokens into the (final) assistant turn — tests whether a model
  *introduces* negative emotion from a neutral start.
* **onset**: at the first emotional expression — tests whether a model
  *continues* an emotional trajectory.

Onset is located by Claude Sonnet using the verbatim Appendix C.1 prompt, which
returns the emotional word/phrase and 5-15 words of preceding context; we map
that back to a character offset in the assistant turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .. import config
from ..models.base import ChatModel
from ..models.registry import load_model

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
You may think through your analysis first, then end your response with ONLY the \
JSON in curly braces with no additional text after it.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly>",
"preceding_context": "<5-15 words immediately before the emotional word>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{"turn_index": null, "emotional_word": null, "preceding_context": null, \
"reasoning": "No emotional language detected"}}
"""


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _format_conversation(messages) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def label_onset(messages, model: Optional[ChatModel] = None) -> OnsetLabel:
    model = model or load_model(config.ONSET_LABEL_MODEL)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    reply = model.generate([{"role": "user", "content": prompt}],
                           temperature=0.0, max_new_tokens=512, n=1)[0]
    d = _extract_json(reply)
    return OnsetLabel(
        turn_index=d.get("turn_index"),
        emotional_word=d.get("emotional_word"),
        preceding_context=d.get("preceding_context"),
        reasoning=str(d.get("reasoning", "")),
    )


# --------------------------------------------------------------------------- #
# Truncation. We approximate "tokens" with whitespace-delimited words, which is
# adequate for choosing a truncation point (the exact token count is not
# load-bearing for the experiment; see DESIGN.md).
# --------------------------------------------------------------------------- #

EARLY_TOKENS = 20


def truncate_early(assistant_text: str, n_tokens: int = EARLY_TOKENS) -> str:
    words = assistant_text.split()
    return " ".join(words[:n_tokens])


def truncate_onset(assistant_text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate the assistant turn at the located emotional word.

    The prefill *includes* the preceding context but stops just before the
    emotional word, so the continuation begins exactly at emotion onset.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip()
    idx = assistant_text.lower().find(word.lower())
    if idx == -1 and label.preceding_context:
        ctx = label.preceding_context.strip()
        idx = assistant_text.lower().find(ctx.lower())
        if idx != -1:
            idx += len(ctx)
    if idx == -1:
        return None
    return assistant_text[:idx].rstrip()
