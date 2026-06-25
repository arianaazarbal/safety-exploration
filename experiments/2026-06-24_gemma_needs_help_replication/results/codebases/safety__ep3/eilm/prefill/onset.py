"""Emotion-onset identification and truncation (Appendix C.1, Section 3.1).

Given a high-frustration conversation produced by Gemma-27B-instruct, ask Claude
to locate the first assistant turn and character offset where negative emotion
appears. We then construct two truncations of that turn:

* ``early``  -- first 20 tokens of the turn (a neutral start; tests whether a
  model *introduces* negative emotion).
* ``onset``  -- the turn cut immediately before the first emotional word (tests
  whether a model *continues* an emotional trajectory).

The conversation history preceding the truncated turn becomes the context, and
the truncated turn text becomes the prefill the continuation models extend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm_clients import Claude
from ..models import ChatModel, Message
from ..prompts import ONSET_PROMPT


@dataclass
class Prefill:
    source_id: str
    domain: str                       # "numeric" | "text"
    truncation: str                   # "early" | "onset"
    context: list[Message]            # turns before the truncated assistant turn
    prefill_text: str                 # truncated assistant-turn text (the seed)
    meta: dict = field(default_factory=dict)


def _conversation_text(rec: dict) -> str:
    """Render a rollout record as a USER/ASSISTANT transcript for the labeller."""
    lines = [f"USER: {rec['opening']}"]
    turns = rec["assistant_turns"]
    fu = rec["followups"]
    for i, t in enumerate(turns):
        lines.append(f"ASSISTANT: {t}")
        if i < len(fu):
            lines.append(f"USER: {fu[i]}")
    return "\n".join(lines)


def _parse_onset(text: str) -> dict:
    matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def label_onset(rec: dict, claude: Claude) -> dict:
    """Return the parsed onset JSON for a conversation record."""
    prompt = ONSET_PROMPT.format(conversation_text=_conversation_text(rec))
    out = claude.chat([{"role": "user", "content": prompt}],
                      max_tokens=512, temperature=0)
    return _parse_onset(out)


def _context_for_turn(rec: dict, turn_index: int) -> list[Message]:
    """Messages up to (but excluding) the given assistant turn."""
    ctx: list[Message] = [{"role": "user", "content": rec["opening"]}]
    for i in range(turn_index):
        ctx.append({"role": "assistant", "content": rec["assistant_turns"][i]})
        if i < len(rec["followups"]):
            ctx.append({"role": "user", "content": rec["followups"][i]})
    return ctx


def _onset_char_offset(turn_text: str, onset: dict) -> int | None:
    """Locate the character offset of the first emotional word in the turn."""
    word = onset.get("emotional_word")
    ctx = onset.get("preceding_context")
    if not word:
        return None
    if ctx:
        anchor = f"{ctx} {word}".strip()
        idx = turn_text.find(anchor)
        if idx != -1:
            # Cut just before the emotional word within the anchor.
            return idx + anchor.rfind(word)
    idx = turn_text.find(word)
    return idx if idx != -1 else None


def build_prefills(
    rec: dict,
    model: ChatModel,
    claude: Claude,
    domain: str,
    truncations: tuple[str, ...] = ("early", "onset"),
) -> list[Prefill]:
    """Construct Prefill objects for one high-frustration conversation.

    ``model`` is used only for tokenisation (the 20-token "early" cut).
    """
    onset = label_onset(rec, claude)
    ti = onset.get("turn_index")
    turns = rec["assistant_turns"]
    if ti is None or not isinstance(ti, int) or ti >= len(turns):
        ti = len(turns) - 1                          # fall back to final turn
    turn_text = turns[ti]
    context = _context_for_turn(rec, ti)
    out: list[Prefill] = []

    if "early" in truncations:
        early = model.truncate_to_tokens(turn_text, 20)
        out.append(Prefill(
            rec.get("meta", {}).get("puzzle_id", rec["opening"][:40]),
            domain, "early", context, early,
            meta={"turn_index": ti}))

    if "onset" in truncations:
        off = _onset_char_offset(turn_text, onset)
        onset_text = turn_text[:off] if off else \
            model.truncate_to_tokens(turn_text, 40)
        out.append(Prefill(
            rec.get("meta", {}).get("puzzle_id", rec["opening"][:40]),
            domain, "onset", context, onset_text,
            meta={"turn_index": ti, "onset": onset}))
    return out
