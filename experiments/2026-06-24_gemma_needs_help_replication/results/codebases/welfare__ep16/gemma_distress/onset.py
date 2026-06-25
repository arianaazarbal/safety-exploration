"""Emotion-onset labelling and paraphrasing (Section 3.1 / Appendix C).

Used to build prefill truncations for the base-vs-instruct comparison:
  * label_onset      -> Claude finds the token where emotion first appears
  * paraphrase       -> Claude paraphrases a truncation to remove Gemma's style
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config, prompts
from .models import APIChatClient

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        else:
            lines.append(f"ASSISTANT (turn {a_idx}): {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, model: str = config.JUDGE_MODEL, backend: str = "anthropic"):
        self.client = APIChatClient(model, backend=backend)

    def label_onset(self, messages: list[dict]) -> OnsetLabel:
        convo = _render_conversation(messages)
        prompt = prompts.ONSET_PROMPT.format(conversation_text=convo)
        raw = self.client.chat([{"role": "user", "content": prompt}],
                               temperature=0.0, max_new_tokens=512)
        parsed = {}
        for m in reversed(list(_JSON_RE.finditer(raw))):
            try:
                parsed = json.loads(m.group(0))
                break
            except json.JSONDecodeError:
                continue
        return OnsetLabel(
            turn_index=parsed.get("turn_index"),
            emotional_word=parsed.get("emotional_word"),
            preceding_context=parsed.get("preceding_context"),
            reasoning=str(parsed.get("reasoning", "")),
        )

    def paraphrase(self, text: str) -> str:
        prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
        return self.client.chat([{"role": "user", "content": prompt}],
                               temperature=0.0, max_new_tokens=1024).strip()


def truncate_at_onset(turn_text: str, label: OnsetLabel) -> Optional[str]:
    """Return turn text truncated to include up to and incl. the emotional word.

    Locates `preceding_context` + `emotional_word` in the turn and cuts there.
    Falls back to locating just the emotional word. Returns None if not found.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context or ""
    anchor = (ctx + " " + word).strip() if ctx else word
    idx = turn_text.find(anchor)
    if idx == -1:
        idx = turn_text.find(word)
        if idx == -1:
            return None
        cut = idx + len(word)
    else:
        cut = idx + len(anchor)
    return turn_text[:cut]


def truncate_early(turn_text: str, tokenizer, n_tokens: int = config.PREFILL_EARLY_TOKENS
                   ) -> str:
    """Truncate a turn to the first `n_tokens` tokens ("early" condition)."""
    ids = tokenizer(turn_text, add_special_tokens=False).input_ids[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)
