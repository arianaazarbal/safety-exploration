"""Emotion-onset labelling, truncation, and paraphrasing (Section 3.1 / App. C).

Three steps that turn a high-frustration Gemma-instruct conversation into prefills:

  1. `label_onset`   - Claude-Sonnet finds where negative emotion first appears.
  2. `truncate_*`    - cut the target assistant turn either 20 tokens in ("early") or
                       at the emotion onset ("onset").
  3. `paraphrase`    - Claude-Sonnet rewrites the truncated text to remove Gemma's
                       stylistic fingerprint (so base/instruct comparisons aren't
                       confounded by surface style).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..models.base import ModelBackend
from ..prompts.judge_prompts import ONSET_PROMPT, PARAPHRASE_PROMPT
from ..utils.llm_clients import make_client


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, cfg: Config):
        jc = cfg.judges["onset"]
        self.client = make_client(jc["provider"], jc["model"],
                                  temperature=jc.get("temperature", 0.0),
                                  max_tokens=jc.get("max_tokens", 1024))

    def label(self, messages: list[dict]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
        out = self.client.chat([{"role": "user", "content": prompt}]).text
        obj = _last_json(out) or {}
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=str(obj.get("reasoning", "")),
        )


class Paraphraser:
    def __init__(self, cfg: Config):
        jc = cfg.judges["paraphrase"]
        self.client = make_client(jc["provider"], jc["model"],
                                  temperature=jc.get("temperature", 0.0),
                                  max_tokens=jc.get("max_tokens", 1024))

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.client.chat([{"role": "user", "content": prompt}]).text.strip()


def truncate_early(backend: ModelBackend, assistant_text: str, n_tokens: int) -> str:
    return backend.truncate_tokens(assistant_text, n_tokens)


def truncate_onset(assistant_text: str, label: OnsetLabel) -> Optional[str]:
    """Cut the assistant turn just before the first emotional word."""
    if not label.emotional_word:
        return None
    ctx = (label.preceding_context or "").strip()
    word = label.emotional_word.strip()
    # Prefer cutting at the emotional word that follows the preceding context.
    if ctx:
        m = re.search(re.escape(ctx), assistant_text)
        if m:
            after = assistant_text[m.end():]
            wm = re.search(re.escape(word), after, re.IGNORECASE)
            cut = m.end() + (wm.start() if wm else 0)
            return assistant_text[:cut].rstrip()
    wm = re.search(re.escape(word), assistant_text, re.IGNORECASE)
    if wm:
        return assistant_text[: wm.start()].rstrip()
    return None


def truncate_before_end(backend: ModelBackend, assistant_text: str, n_tokens: int) -> str:
    """Cut `n_tokens` before the end of the turn (used by the recovery experiment)."""
    total = backend.count_tokens(assistant_text)
    keep = max(0, total - n_tokens)
    return backend.truncate_tokens(assistant_text, keep)


def _last_json(text: str) -> Optional[dict]:
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    spans, depth, start = [], 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(cleaned[start : i + 1])
    for blob in reversed(spans):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None
