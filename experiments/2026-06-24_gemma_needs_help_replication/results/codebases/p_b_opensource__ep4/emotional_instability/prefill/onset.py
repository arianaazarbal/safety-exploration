"""Emotion-onset identification (Section 3.1, Appendix C.1).

Claude Sonnet labels the first point in a conversation where the assistant
expresses negative emotion, returning the emotional word, its preceding context,
and the assistant-turn index. We then locate that point as a character offset in
the corresponding assistant turn, which the truncation step uses to cut the
response just before emotion appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import LABELLER, JudgeSpec
from ..eval.datatypes import ConversationRecord
from ..eval.judge import _extract_json
from ..models import get_backend
from ..models.base import ChatMessage, SamplingParams
from ..prompts.judge_prompts import ONSET_PROMPT_TEMPLATE
from .._util_models import judge_spec_to_modelspec

_PARAMS = SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=1024)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""
    char_offset: Optional[int] = None   # offset within the assistant turn text.


def _render_conversation(record: ConversationRecord) -> str:
    """Render a record as alternating USER/ASSISTANT text for the labeller.

    Assistant turns are numbered from 0 so the returned `turn_index` maps onto
    `record.turns`.
    """
    lines = []
    a = 0
    for turn in record.turns:
        lines.append(f"USER: {turn.user}")
        lines.append(f"ASSISTANT (turn {a}): {turn.assistant}")
        a += 1
    return "\n".join(lines)


def _locate_offset(turn_text: str, preceding: Optional[str], word: Optional[str]) -> Optional[int]:
    """Find the character offset in `turn_text` at which emotion begins.

    Prefer the end of `preceding_context`; fall back to the start of the
    emotional word. Returns None if neither can be located.
    """
    if preceding:
        idx = turn_text.find(preceding)
        if idx != -1:
            return idx + len(preceding)
    if word:
        idx = turn_text.find(word)
        if idx != -1:
            return idx
    return None


class OnsetLabeller:
    def __init__(self, spec: JudgeSpec = LABELLER):
        self.backend = get_backend(judge_spec_to_modelspec(spec, "onset"))

    def label(self, record: ConversationRecord) -> OnsetLabel:
        prompt = ONSET_PROMPT_TEMPLATE.format(
            conversation_text=_render_conversation(record)
        )
        out = self.backend.generate([ChatMessage("user", prompt)], _PARAMS)
        obj = _extract_json(out.text) or {}
        ti = obj.get("turn_index")
        word = obj.get("emotional_word")
        prec = obj.get("preceding_context")
        offset = None
        if isinstance(ti, int) and 0 <= ti < len(record.turns):
            offset = _locate_offset(record.turns[ti].assistant, prec, word)
        return OnsetLabel(
            turn_index=ti if isinstance(ti, int) else None,
            emotional_word=word,
            preceding_context=prec,
            reasoning=str(obj.get("reasoning", "")),
            char_offset=offset,
        )
