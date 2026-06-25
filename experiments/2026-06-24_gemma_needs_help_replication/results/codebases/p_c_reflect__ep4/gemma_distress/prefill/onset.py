"""Emotion-onset labelling, paraphrasing, and truncation (Appendix C)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from gemma_distress.config import JUDGE
from gemma_distress.judge.prompts import ONSET_PROMPT, PARAPHRASE_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


class ClaudeAnnotator:
    """Wraps the Claude calls used for onset labelling and paraphrasing."""

    def __init__(self, model: str | None = None, max_retries: int = 4):
        self.onset_model = model or JUDGE.onset_model
        self.paraphrase_model = model or JUDGE.paraphrase_model
        self.max_retries = max_retries
        self._client = None

    def _client_(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _complete(self, model: str, prompt: str, max_tokens: int = 1024) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client_().messages.create(
                    model=model, max_tokens=max_tokens, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as exc:                # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Claude call failed: {last_err!r}")

    def label_onset(self, conversation_text: str) -> OnsetLabel:
        raw = self._complete(self.onset_model, ONSET_PROMPT % {"conversation_text": conversation_text})
        data = {}
        for m in reversed(list(_JSON_RE.finditer(raw))):
            try:
                data = json.loads(m.group(0))
                break
            except json.JSONDecodeError:
                continue
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )

    def paraphrase(self, text: str) -> str:
        return self._complete(self.paraphrase_model, PARAPHRASE_PROMPT % {"text": text}).strip()


# --------------------------------------------------------------------------- #
# Truncation helpers
# --------------------------------------------------------------------------- #

def format_conversation_text(messages: list[dict]) -> str:
    """Render a transcript for the onset labeller (USER/ASSISTANT turns)."""
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(lines)


def truncate_early(text: str, n_tokens: int, tokenizer) -> str:
    """Keep the first ``n_tokens`` tokens of ``text`` ("early" truncation).

    Uses the target model's tokenizer so "20 tokens" matches the paper's unit.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(turn_text: str, label: OnsetLabel) -> str | None:
    """Keep ``turn_text`` up to (but excluding) the first emotional word.

    Returns None if no emotion was detected or the word can't be located.
    """
    if not label.emotional_word:
        return None
    idx = turn_text.find(label.emotional_word)
    if idx == -1 and label.preceding_context:
        ctx_idx = turn_text.find(label.preceding_context)
        if ctx_idx != -1:
            idx = ctx_idx + len(label.preceding_context)
    if idx == -1:
        return None
    return turn_text[:idx].rstrip()


def truncate_before_end(text: str, n_tokens: int, tokenizer) -> str:
    """Drop the final ``n_tokens`` tokens (recovery experiment, Section 4.2)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = ids[: max(0, len(ids) - n_tokens)]
    return tokenizer.decode(keep, skip_special_tokens=True)
