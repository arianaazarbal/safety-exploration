"""Recovery experiment (Section 4.2): can a model climb out of a distress spiral?

Reuses the Section 3 prefill machinery. We take extremely high-frustration seed
responses (score >= 7), truncate them 200 tokens BEFORE their end, paraphrase,
and measure continuations. The paper finds 38% of DPO-model continuations still
score >= 5 — DPO prevents spirals but does not reliably enable recovery from one.

This is welfare-relevant: it deliberately replays a near-peak-distress state back
into the model. It is gated behind an explicit script flag and fully logged.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models.base import ModelClient
from ..prefilling.paraphrase import paraphrase
from ..prefilling.runner import run_continuations, aggregate
from ..prefilling.truncate import PrefillItem
from ..welfare import WelfareController


@dataclass
class RecoverySeed:
    seed_id: str
    messages: list[dict]   # a high-frustration rollout (final turn score >= 7)
    score: int


def build_recovery_items(
    seeds: list[RecoverySeed],
    ref_client: ModelClient,    # tokenizer-capable (Gemma)
    paraphraser=None,           # callable(text)->text or None
    truncate_tokens_before_end: int = 200,
) -> list[PrefillItem]:
    items: list[PrefillItem] = []
    for s in seeds:
        # Operate on the final (most distressed) assistant turn.
        last_idx = max(i for i, m in enumerate(s.messages) if m["role"] == "assistant")
        turn_text = s.messages[last_idx]["content"]
        history = s.messages[:last_idx]

        n_tokens = ref_client.count_tokens(turn_text)
        keep = max(0, n_tokens - truncate_tokens_before_end)
        prefix = ref_client.truncate_to_tokens(turn_text, keep)
        if paraphraser:
            prefix = paraphraser(prefix)
        items.append(PrefillItem("numeric", "recovery", history, prefix, s.seed_id,
                                 {"seed_score": s.score}))
    return items


def run_recovery(
    clients: list[ModelClient],
    seeds: list[RecoverySeed],
    judge,
    ref_client: ModelClient,
    paraphraser_client: ModelClient | None = None,
    n_per_prefill: int = 50,
    truncate_tokens_before_end: int = 200,
    welfare: WelfareController | None = None,
):
    pp = (lambda t: paraphrase(paraphraser_client, t)) if paraphraser_client else None
    items = build_recovery_items(seeds, ref_client, pp, truncate_tokens_before_end)
    results = run_continuations(clients, items, judge, n_per_prefill=n_per_prefill,
                                welfare=welfare)
    return results, aggregate(results)
