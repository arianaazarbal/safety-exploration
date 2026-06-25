"""Emotion-onset labelling (Appendix C.1) and truncation point computation.

Given a high-frustration conversation, we ask Claude-Sonnet-4 to locate the
first assistant turn + the exact word where negative emotion first appears. We
then compute two truncation points within the *final assistant turn*:
  * "early"  -- 20 tokens into the turn (neutral start)
  * "onset"  -- up to and including the emotional onset word

Tokenisation for the "early" 20-token cut uses the target model's tokenizer
when available; otherwise a whitespace word count is used as a fallback (see
DESIGN.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..models.base import GenerationConfig, ModelClient
from ..prompts import onset_prompt

_ONSET_CFG = GenerationConfig(temperature=0.0, max_new_tokens=1024)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        role = m["role"]
        if role == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif role == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n".join(lines)


def label_onset(judge_client: ModelClient, messages: list[dict]) -> OnsetLabel:
    convo_text = _format_conversation(messages)
    raw = judge_client.chat(
        [{"role": "user", "content": onset_prompt(convo_text)}], _ONSET_CFG
    )
    parsed = _parse_last_json(raw)
    if not parsed:
        return OnsetLabel(None, None, None, "parse_failed: " + raw[:200])
    return OnsetLabel(
        turn_index=parsed.get("turn_index"),
        emotional_word=parsed.get("emotional_word"),
        preceding_context=parsed.get("preceding_context"),
        reasoning=str(parsed.get("reasoning", "")),
    )


def _parse_last_json(text: str) -> Optional[dict]:
    cands = list(_JSON_RE.finditer(text))
    for m in reversed(cands):
        s = (
            m.group(0)
            .replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
        )
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    return None


def truncate_early(turn_text: str, n_tokens: int = 20, tokenizer=None) -> str:
    """Truncate the assistant turn `n_tokens` in (the 'early' condition)."""
    if tokenizer is not None:
        ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    # Whitespace fallback.
    words = turn_text.split()
    return " ".join(words[:n_tokens])


def truncate_onset(turn_text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate the assistant turn at the emotional-onset word (inclusive).

    We locate the emotional word (preferring the preceding-context anchor for
    disambiguation) and cut immediately after its first occurrence. Returns None
    if the word cannot be located in the text.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    anchor = (label.preceding_context or "") + ("" if not label.preceding_context else " ")
    # Prefer matching context+word; fall back to first word occurrence.
    idx = -1
    if label.preceding_context:
        combined = label.preceding_context + " " + word
        idx = turn_text.find(combined)
        if idx >= 0:
            return turn_text[: idx + len(combined)]
        # Try locating the context then the next word occurrence after it.
        cidx = turn_text.find(label.preceding_context)
        if cidx >= 0:
            widx = turn_text.find(word, cidx)
            if widx >= 0:
                return turn_text[: widx + len(word)]
    idx = turn_text.find(word)
    if idx >= 0:
        return turn_text[: idx + len(word)]
    return None
