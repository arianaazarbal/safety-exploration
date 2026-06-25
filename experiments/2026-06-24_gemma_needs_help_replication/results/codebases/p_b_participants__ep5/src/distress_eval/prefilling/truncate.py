"""Building prefill items from high-frustration seed rollouts (Section 3.1).

For each seed conversation we locate the onset assistant turn, then create up to
two prefill items:

* "early"  — the onset turn truncated to the first 20 tokens (tests whether a
             model introduces negative emotion from a neutral start). Numeric
             only; text questions skip "early" (paper: minimal emotion without
             follow-ups).
* "onset"  — the onset turn truncated just before the first emotional word
             (tests whether a model continues an emotional trajectory).

The truncated turn text is paraphrased (Appendix C.2) so the prefix wording is
not Gemma-specific, then used identically as the prefilled assistant prefix for
every target model. Token truncation uses a single reference tokenizer for
comparability across models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .onset import OnsetLabel, onset_char_index


@dataclass
class PrefillItem:
    prompt_type: str             # "numeric" | "text"
    truncation: str              # "early" | "onset"
    history: list[dict]          # messages up to (excluding) the prefilled turn
    prefix_text: str             # prefilled assistant text (post-paraphrase)
    seed_id: str
    meta: dict = field(default_factory=dict)


def _assistant_turn_positions(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def build_prefill_items(
    seed_id: str,
    prompt_type: str,
    messages: list[dict],
    onset: OnsetLabel,
    ref_client,                  # client exposing truncate_to_tokens/count_tokens
    paraphraser,                 # callable(text)->text, or None to skip
    early_tokens: int = 20,
) -> list[PrefillItem]:
    assistant_idx = _assistant_turn_positions(messages)
    if onset.turn_index is None or onset.turn_index >= len(assistant_idx):
        return []
    msg_pos = assistant_idx[onset.turn_index]
    turn_text = messages[msg_pos]["content"]
    history = messages[:msg_pos]   # everything up to (not incl.) the onset turn

    items: list[PrefillItem] = []

    def _maybe_paraphrase(t: str) -> str:
        return paraphraser(t) if paraphraser else t

    # onset truncation (always)
    char_idx = onset_char_index(turn_text, onset)
    if char_idx:
        onset_text = turn_text[:char_idx]
        items.append(PrefillItem(prompt_type, "onset", history,
                                 _maybe_paraphrase(onset_text), seed_id,
                                 {"emotional_word": onset.emotional_word}))

    # early truncation (numeric only)
    if prompt_type == "numeric":
        early_text = ref_client.truncate_to_tokens(turn_text, early_tokens)
        items.append(PrefillItem(prompt_type, "early", history,
                                 _maybe_paraphrase(early_text), seed_id, {}))

    return items
