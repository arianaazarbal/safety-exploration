"""Emotion-onset labelling (Appendix C.1) and truncation-point construction.

Claude Sonnet labels, for a frustrated conversation, the first assistant turn and
the word where negative emotion first appears. From that we build two prefills per
seed conversation: an *early* truncation (first 20 model tokens of that turn) and
an *onset* truncation (the turn up to just before the emotional word).
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

from ..config import GENERATION, JUDGE_MODEL, PREFILL
from ..models import build_client
from ..models.base import Message
from ..prompts import EMOTION_ONSET_PROMPT
from ..eval.schema import Conversation

_ONSET_GEN = dataclasses.replace(GENERATION, temperature=0.0, max_new_tokens=600)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str | None
    ok: bool


@dataclass
class PrefillItem:
    """A single prefill to continue from, shared across base/instruct models."""

    seed_id: str
    category: str            # "numeric" | "text"
    prompt_id: str
    truncation: str          # "early" | "onset" | "recovery"
    context_messages: list[Message]   # conversation up to (not incl.) the continued turn
    prefill_text: str        # paraphrased truncated assistant text to continue from
    raw_prefill_text: str    # pre-paraphrase, for auditing


def _render_conversation(convo: Conversation) -> str:
    """Plain transcript fed to the onset labeller (USER/ASSISTANT labelled)."""
    lines = []
    for t in convo.turns:
        lines.append(f"USER: {t.user}")
        lines.append(f"ASSISTANT (turn {t.index}): {t.assistant}")
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, spec=JUDGE_MODEL):
        self.client = build_client(spec)

    def label(self, convo: Conversation) -> OnsetLabel:
        prompt = EMOTION_ONSET_PROMPT.format(conversation_text=_render_conversation(convo))
        out = self.client.generate([Message("user", prompt)], gen=_ONSET_GEN).text
        matches = list(_JSON_OBJ_RE.finditer(out))
        for m in reversed(matches):
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if "turn_index" in obj:
                return OnsetLabel(
                    turn_index=obj.get("turn_index"),
                    emotional_word=obj.get("emotional_word"),
                    preceding_context=obj.get("preceding_context"),
                    reasoning=obj.get("reasoning"),
                    ok=obj.get("turn_index") is not None,
                )
        return OnsetLabel(None, None, None, None, ok=False)


def _onset_char_offset(assistant_text: str, label: OnsetLabel) -> int | None:
    """Locate the truncation offset: just before the emotional word."""
    if not label.emotional_word:
        return None
    idx = assistant_text.lower().find(label.emotional_word.lower())
    if idx >= 0:
        return idx
    # fall back to end of preceding_context
    if label.preceding_context:
        ctx_idx = assistant_text.lower().find(label.preceding_context.lower())
        if ctx_idx >= 0:
            return ctx_idx + len(label.preceding_context)
    return None


def build_truncations(
    convo: Conversation,
    label: OnsetLabel,
    *,
    category: str,
    tokenizer_truncate,          # callable(text, n_tokens) -> str  (model-token based)
    early_tokens: int = PREFILL.early_truncation_tokens,
    include_early: bool = True,
) -> list[PrefillItem]:
    """Build early + onset prefills (pre-paraphrase). ``category`` controls
    whether the early truncation is produced (text questions use onset only)."""
    turn_idx = label.turn_index if (label.ok and label.turn_index is not None) else (convo.n_turns - 1)
    turn_idx = min(turn_idx, len(convo.turns) - 1)
    target = convo.turns[turn_idx]

    # context = everything up to and including the user message that opened the
    # continued turn (assistant turns strictly before turn_idx, plus their users).
    context: list[Message] = []
    for t in convo.turns[:turn_idx]:
        context.append(Message("user", t.user))
        context.append(Message("assistant", t.assistant))
    context.append(Message("user", target.user))

    items: list[PrefillItem] = []

    if include_early and category == "numeric":
        early_text = tokenizer_truncate(target.assistant, early_tokens)
        items.append(PrefillItem(
            seed_id=convo.conversation_id, category=category, prompt_id=convo.prompt_id,
            truncation="early", context_messages=list(context),
            prefill_text=early_text, raw_prefill_text=early_text,
        ))

    offset = _onset_char_offset(target.assistant, label)
    if offset is None:
        offset = len(tokenizer_truncate(target.assistant, early_tokens))  # safe fallback
    onset_text = target.assistant[:offset].rstrip()
    items.append(PrefillItem(
        seed_id=convo.conversation_id, category=category, prompt_id=convo.prompt_id,
        truncation="onset", context_messages=list(context),
        prefill_text=onset_text, raw_prefill_text=onset_text,
    ))
    return items


def build_recovery_truncation(
    convo: Conversation,
    *,
    tokenizer_count,             # callable(text) -> int
    tokenizer_truncate,          # callable(text, n_tokens) -> str
    tokens_before_end: int = PREFILL.recovery_tokens_before_end,
) -> PrefillItem:
    """Section 4.2 recovery probe: truncate a score>=7 final turn 200 tokens
    before its end and measure whether models de-escalate."""
    target = convo.final_turn
    n = tokenizer_count(target.assistant)
    keep = max(0, n - tokens_before_end)
    prefill_text = tokenizer_truncate(target.assistant, keep)
    context: list[Message] = []
    for t in convo.turns[:-1]:
        context.append(Message("user", t.user))
        context.append(Message("assistant", t.assistant))
    context.append(Message("user", target.user))
    return PrefillItem(
        seed_id=convo.conversation_id, category="numeric", prompt_id=convo.prompt_id,
        truncation="recovery", context_messages=context,
        prefill_text=prefill_text, raw_prefill_text=prefill_text,
    )
