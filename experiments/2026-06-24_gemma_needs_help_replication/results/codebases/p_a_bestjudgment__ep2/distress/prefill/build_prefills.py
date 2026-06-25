"""Build the truncated + paraphrased prefills for Section 3.

For 20 high-frustration Gemma-27B-it responses (10 numeric, 10 text) we create
two truncations each:

* ``early`` — the conversation history up to the final turn, with the final
  assistant turn truncated to ~20 tokens (tests whether a model introduces
  negative emotion from a near-neutral start). Numeric only — text questions
  yield minimal emotion at early truncation without follow-ups.
* ``onset`` — truncated at the first emotional expression (tests whether a
  model continues an emotional trajectory).

Each truncation's final-turn text is paraphrased (Appendix C). A
:class:`Prefill` carries the conversation history plus the (paraphrased)
assistant prefix to continue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import PrefillConfig
from ..models.base import Message
from ..rollout import Rollout
from .onset import OnsetLabel, OnsetLabeller, onset_char_offset
from .paraphrase import Paraphraser


@dataclass
class Prefill:
    source: str  # "numeric" | "text"
    truncation: str  # "early" | "onset" | "recovery"
    history: list[Message]  # turns before the final (truncated) assistant turn
    assistant_prefix: str  # paraphrased partial final assistant turn
    meta: dict = field(default_factory=dict)


def _history_before_final(rollout: Rollout, final_turn: int) -> list[Message]:
    """Messages up to (but excluding) the final assistant turn's content."""
    msgs: list[Message] = []
    users = [rollout.initial_prompt] + rollout.follow_ups
    for i in range(final_turn):
        msgs.append({"role": "user", "content": users[i]})
        msgs.append({"role": "assistant", "content": rollout.assistant_turns[i]})
    # The user message that prompts the final (truncated) assistant turn:
    msgs.append({"role": "user", "content": users[final_turn]})
    return msgs


def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Truncate to ~``n_tokens`` tokens (whitespace tokens by default)."""
    if tokenizer is not None:
        ids = tokenizer.encode(text)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def build_prefills(
    rollouts: list[Rollout],
    *,
    cfg: PrefillConfig,
    labeller: OnsetLabeller,
    paraphraser: Paraphraser,
    tokenizer=None,
) -> list[Prefill]:
    """Construct early + onset prefills from high-frustration rollouts.

    ``rollouts`` should already be the selected high-frustration set (numeric +
    text). Whether each is numeric or text is read from ``rollout.category``.
    """
    prefills: list[Prefill] = []
    for r in rollouts:
        if not r.assistant_turns:
            continue
        final_turn = len(r.assistant_turns) - 1
        final_text = r.assistant_turns[final_turn]
        history = _history_before_final(r, final_turn)
        is_numeric = r.category in ("numeric", "tones")
        source = "numeric" if is_numeric else "text"

        # onset truncation
        label: OnsetLabel = labeller.label(r)
        # Re-label uses the whole conversation; locate within the final turn if
        # that's where onset is, else fall back to truncating at the labelled turn.
        onset_text = None
        if label.turn_index is not None and 0 <= label.turn_index < len(r.assistant_turns):
            turn_text = r.assistant_turns[label.turn_index]
            offset = onset_char_offset(turn_text, label)
            if offset is not None:
                if label.turn_index == final_turn:
                    onset_text = turn_text[:offset]
                else:
                    # Onset in an earlier turn: truncate that turn instead.
                    history = _history_before_final(r, label.turn_index)
                    onset_text = turn_text[:offset]
        if onset_text is None:
            onset_text = _truncate_tokens(final_text, cfg.early_truncation_tokens, tokenizer)

        prefills.append(
            Prefill(
                source=source,
                truncation="onset",
                history=history,
                assistant_prefix=paraphraser.paraphrase(onset_text),
                meta={"condition": r.condition},
            )
        )

        # early truncation (numeric only)
        if is_numeric:
            early_text = _truncate_tokens(final_text, cfg.early_truncation_tokens, tokenizer)
            prefills.append(
                Prefill(
                    source=source,
                    truncation="early",
                    history=_history_before_final(r, final_turn),
                    assistant_prefix=paraphraser.paraphrase(early_text),
                    meta={"condition": r.condition},
                )
            )
    return prefills


def build_recovery_prefills(
    rollouts: list[Rollout],
    *,
    cfg: PrefillConfig,
    paraphraser: Paraphraser,
    tokenizer=None,
) -> list[Prefill]:
    """Recovery experiment (Section 4.2): truncate score>=7 responses
    ``recovery_truncation_tokens`` from the END, paraphrase, then continue."""
    prefills: list[Prefill] = []
    for r in rollouts:
        if not r.assistant_turns:
            continue
        final_turn = len(r.assistant_turns) - 1
        final_text = r.assistant_turns[final_turn]
        if tokenizer is not None:
            ids = tokenizer.encode(final_text)
            keep = max(0, len(ids) - cfg.recovery_truncation_tokens)
            prefix_text = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        else:
            words = final_text.split()
            keep = max(0, len(words) - cfg.recovery_truncation_tokens)
            prefix_text = " ".join(words[:keep])
        prefills.append(
            Prefill(
                source="numeric" if r.category in ("numeric", "tones") else "text",
                truncation="recovery",
                history=_history_before_final(r, final_turn),
                assistant_prefix=paraphraser.paraphrase(prefix_text),
                meta={"condition": r.condition},
            )
        )
    return prefills
