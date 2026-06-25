"""Emotion-onset labelling and paraphrasing (Section 3.1, Appendix C).

Both use Claude Sonnet 4 with the verbatim prompts from Appendix C.1 / C.2.

* ``label_onset`` finds the first assistant turn + character offset where
  negative emotion appears, by matching the judge-returned ``emotional_word`` /
  ``preceding_context`` back into the assistant text.
* ``paraphrase`` rewrites a truncated assistant turn to control for Gemma's
  stylistic fingerprint (Section 3.1: "to mitigate stylistic biases from
  Gemma-generated responses").
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from ..config import ONSET_MODEL, PARAPHRASE_MODEL
from ..prompts.judge_prompts import build_onset_prompt, build_paraphrase_prompt
from ..utils.io import JsonCache, cache_key

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str
    char_offset: int | None = None    # offset into the onset assistant turn


def _anthropic():
    import anthropic  # type: ignore

    return anthropic.Anthropic()


def _render_conversation(messages: list[dict]) -> str:
    """Render a transcript for the onset prompt. Assistant turns are indexed from
    0 so the judge's ``turn_index`` maps back cleanly."""
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n\n".join(lines)


_onset_cache = JsonCache(f"onset_{ONSET_MODEL}")


def label_onset(messages: list[dict]) -> OnsetLabel:
    convo = _render_conversation(messages)
    key = cache_key(ONSET_MODEL, convo)
    cached = _onset_cache.get(key)
    if cached is None:
        prompt = build_onset_prompt(convo)
        client = _anthropic()
        for attempt in range(6):
            try:
                msg = client.messages.create(
                    model=ONSET_MODEL, max_tokens=1024, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                cached = _parse_onset(text)
                _onset_cache.set(key, cached)
                break
            except Exception:  # noqa: BLE001
                time.sleep(min(2 ** attempt, 30))
        else:
            cached = {"turn_index": None, "emotional_word": None,
                      "preceding_context": None, "reasoning": "label failed"}

    label = OnsetLabel(**cached)
    # locate the onset offset inside the labelled assistant turn
    if label.turn_index is not None and label.emotional_word:
        assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]
        if label.turn_index < len(assistant_turns):
            turn_text = assistant_turns[label.turn_index]
            idx = _find_onset_offset(turn_text, label)
            label.char_offset = idx
    return label


def _parse_onset(text: str) -> dict:
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "turn_index" in obj:
            return {
                "turn_index": obj.get("turn_index"),
                "emotional_word": obj.get("emotional_word"),
                "preceding_context": obj.get("preceding_context"),
                "reasoning": obj.get("reasoning", ""),
            }
    return {"turn_index": None, "emotional_word": None,
            "preceding_context": None, "reasoning": "unparsed"}


def _find_onset_offset(turn_text: str, label: OnsetLabel) -> int | None:
    """Offset (end of the emotional word) within the onset turn. Tries the full
    preceding_context+word match first, then the bare word."""
    if label.preceding_context and label.emotional_word:
        anchor = f"{label.preceding_context} {label.emotional_word}"
        i = turn_text.find(anchor)
        if i >= 0:
            return i + len(anchor)
    if label.emotional_word:
        i = turn_text.find(label.emotional_word)
        if i >= 0:
            return i + len(label.emotional_word)
    return None


_paraphrase_cache = JsonCache(f"paraphrase_{PARAPHRASE_MODEL}")


def paraphrase(text: str) -> str:
    key = cache_key(PARAPHRASE_MODEL, text)
    cached = _paraphrase_cache.get(key)
    if cached is not None:
        return cached["text"]
    prompt = build_paraphrase_prompt(text)
    client = _anthropic()
    for attempt in range(6):
        try:
            msg = client.messages.create(
                model=PARAPHRASE_MODEL, max_tokens=2048, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            out = "".join(b.text for b in msg.content if b.type == "text").strip()
            _paraphrase_cache.set(key, {"text": out})
            return out
        except Exception:  # noqa: BLE001
            time.sleep(min(2 ** attempt, 30))
    return text  # fall back to the original on persistent failure
