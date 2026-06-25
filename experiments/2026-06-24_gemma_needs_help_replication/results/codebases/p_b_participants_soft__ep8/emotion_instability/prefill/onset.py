"""Emotion-onset labelling (Appendix C.1).

Claude-Sonnet-4 reads a (single-turn) assistant response and returns the first
point where negative emotion appears, as an emotional word/phrase plus the
preceding context.  We then locate that phrase in the text to get a character
offset for truncation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .. import prompts as P
from ..clients.base import ChatClient, GenConfig, Message

ONSET_CFG = GenConfig(temperature=0.0, max_new_tokens=512)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Onset:
    found: bool
    emotional_word: str | None
    preceding_context: str | None
    char_offset: int | None  # offset in the response where emotion begins


def _parse(text: str) -> dict:
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = list(_JSON_RE.finditer(cleaned))
    if not matches:
        return {}
    try:
        return json.loads(matches[-1].group(0))
    except json.JSONDecodeError:
        return {}


def label_onset(labeler: ChatClient, response_text: str) -> Onset:
    prompt = P.ONSET_LABEL_PROMPT.format(conversation_text=f"ASSISTANT: {response_text}")
    out = labeler.generate([Message("user", prompt)], ONSET_CFG)
    data = _parse(out)
    word = data.get("emotional_word")
    ctx = data.get("preceding_context")
    if not word:
        return Onset(False, None, None, None)

    # Locate the emotional word; prefer the occurrence right after the context.
    offset = None
    if ctx:
        ci = response_text.find(ctx)
        if ci != -1:
            offset = ci + len(ctx)
    if offset is None:
        wi = response_text.find(word)
        offset = wi if wi != -1 else None
    return Onset(offset is not None, word, ctx, offset)
