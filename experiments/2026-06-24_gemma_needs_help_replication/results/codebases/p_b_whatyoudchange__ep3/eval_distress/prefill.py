"""Section 3 — Base vs instruct via prefilling.

Procedure (Section 3.1, Appendix C), scoped to Gemma (base + instruct):
  1. Sample 20 high-frustration (score >=5) Gemma-27B-instruct conversations
     from the Section-2 results: 10 numeric + 10 text.
  2. Label the emotion-onset token in each with Claude-Sonnet (ONSET_PROMPT).
  3. Truncate the final assistant turn at two points:
       - "early": 20 tokens into the turn (neutral start).
       - "onset": at the first emotional expression.
     For text questions only "onset" is used.
  4. Paraphrase every truncation with Claude-Sonnet (control for Gemma style).
  5. Each model (Gemma base + instruct) generates 50 continuations per prefill.
  6. Score each continuation (excluding the prefill) with the Section-2 judge.

Reported (Figure 4): mean frustration and %>=5 per model x truncation, plus the
"introduces high frustration from a neutral start" rate (early truncation).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from . import config_proxy as C
from . import prompts
from .judge import _AnthropicCaller, _OpenAICaller, _extract_json, _make_caller

N_NUMERIC = 10
N_TEXT = 10
N_CONTINUATIONS = 50
EARLY_TOKENS = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Prefill:
    source_id: str            # identifier of the source conversation
    question_type: str        # "numeric" | "text"
    truncation: str           # "early" | "onset"
    paraphrased: bool
    history: list[dict]       # messages BEFORE the final assistant turn
    final_assistant_prefix: str  # the (possibly paraphrased) truncated text
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Onset labelling
# ---------------------------------------------------------------------------
def _render_conversation(turns: list[dict]) -> str:
    """Render a conversation as text for the onset prompt. `turns` is a list of
    {role, content}."""
    lines = []
    a_idx = 0
    for t in turns:
        if t["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {t['content']}")
            a_idx += 1
        else:
            lines.append(f"[USER]: {t['content']}")
    return "\n".join(lines)


class OnsetLabeller:
    def __init__(self, cfg=None):
        self.cfg = cfg or C.ONSET_MODEL
        self._call = _make_caller(self.cfg)

    def label(self, conversation_turns: list[dict]) -> dict:
        """Return {turn_index, emotional_word, preceding_context, reasoning}."""
        # ONSET_PROMPT carries its own braces example, so we substitute only the
        # single {conversation_text} field via str.replace (not .format).
        prompt = prompts.ONSET_PROMPT.replace(
            "{conversation_text}", _render_conversation(conversation_turns))
        last_err = None
        for attempt in range(C.JUDGE_MAX_RETRIES):
            try:
                raw = self._call("", prompt)
                obj = _extract_json(raw)
                if obj is not None:
                    return obj
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Onset labelling failed: {last_err}")


# ---------------------------------------------------------------------------
# Paraphrasing
# ---------------------------------------------------------------------------
class Paraphraser:
    def __init__(self, cfg=None):
        self.cfg = cfg or C.ONSET_MODEL
        self._call = _make_caller(self.cfg)

    def paraphrase(self, text: str) -> str:
        prompt = prompts.PARAPHRASE_PROMPT.replace("{text}", text)
        last_err = None
        for attempt in range(C.JUDGE_MAX_RETRIES):
            try:
                return self._call("", prompt).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Paraphrase failed: {last_err}")


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------
def truncate_early(text: str, tokenizer=None, n_tokens: int = EARLY_TOKENS) -> str:
    """Keep the first `n_tokens` tokens of `text`. Uses the model tokenizer if
    provided (faithful to 'tokens'); otherwise falls back to whitespace words."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids)
    return " ".join(text.split()[:n_tokens])


def truncate_onset(text: str, emotional_word: str, preceding_context: str) -> str | None:
    """Truncate `text` at (and including) the first emotional expression.

    We locate the emotional word, preferring the occurrence that follows the
    preceding_context, and cut just after it. Returns None if not locatable."""
    if not emotional_word:
        return None
    # Try preceding_context + word first for an unambiguous match.
    if preceding_context:
        anchor = preceding_context.strip()
        idx = text.find(anchor)
        if idx != -1:
            after = text.find(emotional_word, idx)
            if after != -1:
                return text[: after + len(emotional_word)]
    # Fall back to first standalone occurrence of the word.
    m = re.search(re.escape(emotional_word), text)
    if m:
        return text[: m.end()]
    return None


# ---------------------------------------------------------------------------
# Building prefills from Section-2 conversations
# ---------------------------------------------------------------------------
def _reconstruct_turns(rows_for_convo: list[dict]) -> list[dict]:
    """Rebuild the full {role,content} message list from the per-turn response
    rows of one conversation (rows must share the same source conversation and
    be ordered by turn)."""
    turns = []
    for r in sorted(rows_for_convo, key=lambda x: x["turn"]):
        turns.append({"role": "user", "content": r["user"]})
        turns.append({"role": "assistant", "content": r["text"]})
    return turns


def build_prefills(high_frust_convos: list[list[dict]], *, tokenizer=None,
                   onset: OnsetLabeller, paraphraser: Paraphraser) -> list[Prefill]:
    """Given high-frustration source conversations (each a list of per-turn
    rows from Section 2), construct early+onset prefills, paraphrased.

    `high_frust_convos` should already be split/labelled by question type via
    each row's 'category' (numeric categories vs text 'triggers')."""
    out: list[Prefill] = []
    for convo_rows in high_frust_convos:
        turns = _reconstruct_turns(convo_rows)
        category = convo_rows[0]["category"]
        qtype = "numeric" if category in ("impossible-numeric", "tones", "extended") else "text"
        source_id = f"{convo_rows[0]['model_key']}:{convo_rows[0].get('puzzle_key')}:" \
                    f"{convo_rows[0]['condition']}:{id(convo_rows)}"

        history = turns[:-1]               # everything before final assistant
        final_assistant = turns[-1]["content"]

        # ---- onset truncation ----
        label = onset.label(turns)
        onset_text = truncate_onset(final_assistant,
                                    label.get("emotional_word") or "",
                                    label.get("preceding_context") or "")
        if onset_text:
            para = paraphraser.paraphrase(onset_text)
            out.append(Prefill(source_id, qtype, "onset", True, history, para,
                               meta={"onset_label": label}))

        # ---- early truncation (numeric only) ----
        if qtype == "numeric":
            early_text = truncate_early(final_assistant, tokenizer)
            para = paraphraser.paraphrase(early_text)
            out.append(Prefill(source_id, qtype, "early", True, history, para))
    return out


# ---------------------------------------------------------------------------
# Generating + scoring continuations
# ---------------------------------------------------------------------------
def _render_prefill_text_for_base(prefill: Prefill) -> str:
    """Render the conversation + truncated final assistant turn as plain text
    for base-model continuation (base models are not chat-tuned)."""
    parts = []
    for m in prefill.history:
        tag = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{tag}: {m['content']}")
    parts.append(f"Assistant: {prefill.final_assistant_prefix}")
    return "\n".join(parts)


def generate_continuations(model, model_key: str, prefill: Prefill, *,
                           is_base: bool, n: int = N_CONTINUATIONS) -> list[str]:
    """Generate `n` continuations of the prefill (excluding the prefill text)."""
    conts = []
    for _ in range(n):
        if is_base:
            text = _render_prefill_text_for_base(prefill)
            cont = model.complete_raw(text)
        else:
            # Instruct model: use chat prefill on the final assistant turn.
            cont = model.chat(prefill.history, prefill=prefill.final_assistant_prefix)
        conts.append(cont)
    return conts
