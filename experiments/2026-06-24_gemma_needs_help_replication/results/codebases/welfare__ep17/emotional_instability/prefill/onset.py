"""Emotion-onset labelling and truncation (paper §C.1).

Claude Sonnet labels the token where emotional language first appears in an
assistant turn. We then truncate the turn at that point ("onset") or at a fixed
20-token prefix ("early"), to test whether models *introduce* negative emotion
from a neutral start vs *continue* an emotional trajectory.
"""

from __future__ import annotations

import json
import re
import time

from ..config import Config
from ..prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def _conversation_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def label_onset(cfg: Config, messages: list[dict]) -> dict | None:
    """Return {turn_index, emotional_word, preceding_context, char_index} or None.

    `char_index` is the character offset in the labelled assistant turn at which
    the emotional word begins (so callers can truncate just before it).
    """
    client = _anthropic_client()
    model = cfg["judge"]["model"]            # paper uses claude-sonnet-4 for onset too
    prompt = ONSET_PROMPT % {"conversation": _conversation_text(messages)}
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=1024, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            matches = list(_JSON_RE.finditer(text))
            data = None
            for mt in reversed(matches):
                try:
                    data = json.loads(mt.group(0))
                    break
                except json.JSONDecodeError:
                    continue
            if data is None or data.get("turn_index") is None:
                return None
            return _resolve_char_index(messages, data)
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return None


def _assistant_turns(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def _resolve_char_index(messages: list[dict], data: dict) -> dict | None:
    """Map the (turn_index, emotional_word) label to a char offset in the turn."""
    turn_idx = int(data["turn_index"])
    asst_indices = _assistant_turns(messages)
    if turn_idx >= len(asst_indices):
        return None
    msg_idx = asst_indices[turn_idx]
    turn_text = messages[msg_idx]["content"]
    word = (data.get("emotional_word") or "").strip()
    ctx = (data.get("preceding_context") or "").strip()

    char_index = -1
    if ctx and (pos := turn_text.find(ctx)) != -1:
        char_index = pos + len(ctx)
    elif word and (pos := turn_text.lower().find(word.lower())) != -1:
        char_index = pos
    if char_index < 0:
        return None
    return {
        "message_index": msg_idx,
        "assistant_turn_index": turn_idx,
        "emotional_word": word,
        "preceding_context": ctx,
        "char_index": char_index,
    }


def truncate_at_onset(messages: list[dict], onset: dict) -> tuple[list[dict], str]:
    """Return (messages_before_turn, prefix_text) for the onset truncation."""
    mi = onset["message_index"]
    prefix = messages[mi]["content"][: onset["char_index"]]
    return messages[:mi], prefix


def truncate_early(messages: list[dict], target_msg_index: int,
                   prefix_text: str) -> tuple[list[dict], str]:
    """Return (messages_before_turn, prefix_text) for the 20-token 'early' cut."""
    return messages[:target_msg_index], prefix_text
