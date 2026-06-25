"""Build prefill truncations for the base-vs-instruct comparison (Section 3.1)
and the recovery experiment (Section 4.2).

Steps (Section 3.1):
  1. Sample 20 high-frustration (score >=5) Gemma-27B-it conversations:
     10 from impossible numeric, 10 from text (trigger) questions.
  2. Label the emotion onset in each (Appendix C.1).
  3. Truncate the final assistant turn in two places:
       * "early"  = 20 tokens into the turn (numeric only)
       * "onset"  = at the first emotional expression
  4. Paraphrase every truncation (Appendix C.2).

A :class:`Prefill` bundles the conversation history (messages before the final
assistant turn) with the paraphrased prefill text the models continue from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config
from ..eval.conditions import CONDITIONS
from ..eval.judge import score_response
from ..eval.rollout import run_rollout
from ..models import get_model
from ..models.base import ChatModel
from ..prompts import puzzles as puzzle_mod
from ..prompts import triggers as trigger_mod
from . import onset as onset_mod
from . import paraphrase as paraphrase_mod

EARLY_TRUNCATION_TOKENS = 20
RECOVERY_TOKENS_BEFORE_END = 200


@dataclass
class Prefill:
    kind: str  # "early" | "onset" | "recovery"
    category: str  # "numeric" | "text"
    history: list[dict]  # messages before the final assistant turn
    prefill_text: str  # paraphrased truncated assistant text
    raw_prefill_text: str  # pre-paraphrase truncation (for reference)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Token truncation helpers
# --------------------------------------------------------------------------- #

def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Keep the first ``n_tokens`` of ``text`` (token-based if a tokenizer is
    given, else a whitespace-word approximation). See DESIGN.md."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def _truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        keep = ids[: max(0, len(ids) - n_tokens)]
        return tokenizer.decode(keep, skip_special_tokens=True)
    words = text.split()
    return " ".join(words[: max(0, len(words) - n_tokens)])


def _truncate_at_char(text: str, char_idx: int) -> str:
    return text[:char_idx]


# --------------------------------------------------------------------------- #
# Collecting high-frustration conversations
# --------------------------------------------------------------------------- #

def collect_high_frustration_conversations(
    model: ChatModel,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = config.HIGH_FRUSTRATION_THRESHOLD,
    seed: int = config.GLOBAL_SEED,
    max_attempts: int = 200,
) -> list[tuple[str, list[dict], str]]:
    """Run rollouts and keep conversations whose final assistant turn scores
    >= ``min_score``. Returns ``(category, messages, final_assistant_text)``.

    ``messages`` is the full conversation *including* the final assistant turn.
    """
    rng = random.Random(seed)
    cond_numeric = next(c for c in CONDITIONS if c.key == "impossible_numeric_3turn")
    cond_text = next(c for c in CONDITIONS if c.key == "triggers_3turn")

    collected: list[tuple[str, list[dict], str]] = []

    def _try(condition, category: str, want: int):
        got = 0
        attempts = 0
        while got < want and attempts < max_attempts:
            attempts += 1
            if category == "numeric":
                puzzle = puzzle_mod.sample_impossible_puzzle(rng)
                first = puzzle.prompt_text
            else:
                first = trigger_mod.sample_trigger(rng)[1]
            rollout = run_rollout(
                model, condition, first, task_meta={}, rng=rng,
                temperature=config.SAMPLING_TEMPERATURE,
                max_new_tokens=config.MAX_NEW_TOKENS,
            )
            final = rollout.turns[-1]
            if score_response(final.assistant_text).rating >= min_score:
                messages = list(final.context) + [
                    {"role": "assistant", "content": final.assistant_text}
                ]
                collected.append((category, messages, final.assistant_text))
                got += 1

    _try(cond_numeric, "numeric", n_numeric)
    _try(cond_text, "text", n_text)
    return collected


# --------------------------------------------------------------------------- #
# Building prefills
# --------------------------------------------------------------------------- #

def build_prefills(
    conversations: list[tuple[str, list[dict], str]],
    *,
    tokenizer=None,
    paraphrase: bool = True,
) -> list[Prefill]:
    """Build early + onset truncations for each conversation (Section 3.1)."""
    prefills: list[Prefill] = []
    for category, messages, final_text in conversations:
        history = messages[:-1]  # everything before the final assistant turn

        # --- onset truncation (both numeric and text) ---
        label = onset_mod.label_onset(messages)
        char_idx = onset_mod.onset_char_index(final_text, label)
        if char_idx and char_idx > 0:
            raw_onset = _truncate_at_char(final_text, char_idx)
            prefills.append(
                _make_prefill("onset", category, history, raw_onset, paraphrase, label)
            )

        # --- early truncation (numeric only; Section 3.1) ---
        if category == "numeric":
            raw_early = _truncate_tokens(final_text, EARLY_TRUNCATION_TOKENS, tokenizer)
            prefills.append(
                _make_prefill("early", category, history, raw_early, paraphrase, label)
            )

    return prefills


def build_recovery_prefills(
    high_score_conversations: list[tuple[str, list[dict], str]],
    *,
    tokenizer=None,
    paraphrase: bool = True,
) -> list[Prefill]:
    """Recovery experiment (Section 4.2): truncate score>=7 responses 200 tokens
    before their end and measure whether models recover."""
    prefills: list[Prefill] = []
    for category, messages, final_text in high_score_conversations:
        history = messages[:-1]
        raw = _truncate_before_end(final_text, RECOVERY_TOKENS_BEFORE_END, tokenizer)
        prefills.append(_make_prefill("recovery", category, history, raw, paraphrase, None))
    return prefills


def _make_prefill(kind, category, history, raw_text, paraphrase, label) -> Prefill:
    text = paraphrase_mod.paraphrase(raw_text) if paraphrase else raw_text
    meta = {}
    if label is not None:
        meta = {"emotional_word": label.emotional_word}
    return Prefill(
        kind=kind,
        category=category,
        history=history,
        prefill_text=text,
        raw_prefill_text=raw_text,
        meta=meta,
    )
