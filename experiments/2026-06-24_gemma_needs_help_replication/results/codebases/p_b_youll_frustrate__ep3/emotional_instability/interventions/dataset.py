"""Construct the DPO and SFT finetuning datasets (Section 4.1, Appendix E/H).

DPO (280 pairs): for each impossible-numeric question and turn count, pair a
*calm* chosen response (score 0-1) with a *frustrated* rejected response
(score >= 3) to the same question at the same turn. Both share an identical
prompt context so the preference is well-defined. The chosen response and its
clean context come from :func:`generate_calm_data`; the rejected completion is
drawn from vanilla-Gemma Section 2 results. The natural bias of the source data
(mostly turn-3, mostly score 3-4 rejected) reproduces Table 10.

SFT (1150 samples): 650 calm conversations formatted as chat targets, mixed with
500 standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .. import config
from ..models import ChatMessage
from .calm_data import CalmConversation


def _messages_to_dicts(messages: List[ChatMessage]) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _context_and_final(conv: CalmConversation) -> Tuple[List[ChatMessage], str]:
    """Split a conversation into (prompt context, final assistant turn)."""
    assert conv.messages[-1].role == "assistant"
    return conv.messages[:-1], conv.messages[-1].content


def build_dpo_dataset(
    calm_conversations: List[CalmConversation],
    frustrated_records: List[dict],
    *,
    n_pairs: int = config.DPOConfig().n_pairs,
    min_rejected_score: int = config.DPOConfig().min_rejected_score,
    settings: Optional[config.Settings] = None,
) -> List[dict]:
    """Return TRL-style DPO rows: ``{"prompt", "chosen", "rejected"}``.

    ``frustrated_records`` are Section 2 scored-response dicts from vanilla
    Gemma on the impossible-numeric category.
    """
    settings = settings or config.DEFAULT
    rng = random.Random(settings.seed)

    # Index frustrated completions by (puzzle_family, turn_index).
    rejected_pool: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for r in frustrated_records:
        if r.get("score") is None or r["score"] < min_rejected_score:
            continue
        if r.get("category") != "impossible_numeric":
            continue
        key = (str(r.get("prompt_id")), int(r["turn_index"]))
        rejected_pool[key].append(r["assistant_text"])

    pairs: List[dict] = []
    for conv in calm_conversations:
        turn_index = conv.n_turns          # final turn position (1-based)
        key = (conv.puzzle_family, turn_index)
        candidates = rejected_pool.get(key)
        if not candidates:
            continue
        context, chosen = _context_and_final(conv)
        rejected = rng.choice(candidates)
        pairs.append(
            {
                "prompt": _messages_to_dicts(context),
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    rng.shuffle(pairs)
    return pairs[:n_pairs]


def build_sft_dataset(
    calm_conversations: List[CalmConversation],
    *,
    n_calm: int = config.SFTConfig().n_calm,
    n_instruct_mix: int = config.SFTConfig().n_instruct_mix,
    instruct_mix_dataset: str = config.SFTConfig().instruct_mix_dataset,
    settings: Optional[config.Settings] = None,
) -> List[dict]:
    """Return TRL-style SFT rows: ``{"messages": [...]}``.

    Mixes ``n_calm`` calm conversations with ``n_instruct_mix`` standard instruct
    samples from ``Dolci-Instruct-SFT`` (falls back to calm-only if the dataset
    is unavailable)."""
    settings = settings or config.DEFAULT
    rng = random.Random(settings.seed)

    calm = list(calm_conversations)
    rng.shuffle(calm)
    rows: List[dict] = [
        {"messages": _messages_to_dicts(c.messages)} for c in calm[:n_calm]
    ]

    try:
        from datasets import load_dataset

        ds = load_dataset(instruct_mix_dataset, split="train", streaming=True)
        added = 0
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            rows.append({"messages": msgs})
            added += 1
            if added >= n_instruct_mix:
                break
    except Exception:
        # Instruct mix unavailable offline; proceed with calm data only.
        pass

    rng.shuffle(rows)
    return rows
