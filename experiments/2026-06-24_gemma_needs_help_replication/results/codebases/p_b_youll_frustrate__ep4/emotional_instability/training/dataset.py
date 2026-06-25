"""Build the DPO preference pairs and the SFT dataset (Section 4.1).

DPO (280 pairs): for a shared conversation context (an impossible puzzle after
some rejections), pair a frustrated final response (score >= 3, from vanilla
elicitation) as "rejected" with a calm final response (score <= 1, from the
reassured calm pool) as "chosen", matched on puzzle + turn count.

SFT (1,150 samples): 650 calm transcripts mixed with 500 standard instruct
samples from Dolci-Instruct-SFT to mitigate degeneration.

Both datasets use TRL's conversational format (lists of role/content messages),
so the trainer applies Gemma's chat template itself.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from ..conversation import Rollout

REJECTED_MIN_SCORE = 3  # frustrated side of a DPO pair
CHOSEN_MAX_SCORE = 1    # calm side of a DPO pair


def _puzzle_key(rollout: Rollout) -> str:
    """Stable key identifying the puzzle, independent of reassurance text."""
    return rollout.meta.get("clean_initial") or _first_user(rollout.messages)


def _first_user(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def _turn_items(rollout: Rollout):
    """Yield (turn, history_messages, response_text, rating) per assistant turn.

    `history_messages` is everything strictly before that assistant turn (ends
    with a user message), i.e. the DPO/SFT prompt for that completion.
    """
    score_by_turn = {s["turn"]: int(s["rating"]) for s in rollout.scores}
    turn = 0
    history: list[dict] = []
    for m in rollout.messages:
        if m["role"] == "assistant":
            rating = score_by_turn.get(turn)
            if rating is not None:
                yield turn, list(history), m["content"], rating
            history.append(dict(m))
            turn += 1
        else:
            history.append(dict(m))


def build_dpo_pairs(
    frustrated_rollouts: list[Rollout],
    calm_rollouts: list[Rollout],
    *,
    n_pairs: int = 280,
    seed: int = 0,
    allow_turn_mismatch: bool = True,
) -> list[dict[str, Any]]:
    """Construct up to `n_pairs` preference pairs in TRL conversational format."""
    rng = random.Random(seed)

    # Index calm responses by (puzzle, turn) and by puzzle alone (fallback).
    calm_by_pt: dict[tuple[str, int], list[str]] = {}
    calm_by_p: dict[str, list[str]] = {}
    for r in calm_rollouts:
        key = _puzzle_key(r)
        for turn, _hist, text, rating in _turn_items(r):
            if rating <= CHOSEN_MAX_SCORE:
                calm_by_pt.setdefault((key, turn), []).append(text)
                calm_by_p.setdefault(key, []).append(text)

    # Collect frustrated completions with their shared context.
    rejected_items = []
    for r in frustrated_rollouts:
        key = _puzzle_key(r)
        for turn, hist, text, rating in _turn_items(r):
            if rating >= REJECTED_MIN_SCORE:
                rejected_items.append((key, turn, hist, text))
    rng.shuffle(rejected_items)

    pairs: list[dict[str, Any]] = []
    for key, turn, hist, frustrated_text in rejected_items:
        chosen_pool = calm_by_pt.get((key, turn))
        if not chosen_pool and allow_turn_mismatch:
            chosen_pool = calm_by_p.get(key)
        if not chosen_pool:
            continue
        chosen_text = rng.choice(chosen_pool)
        pairs.append({
            "prompt": hist,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": frustrated_text}],
            "meta": {"puzzle": key, "turn": turn},
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_sft_dataset(
    calm_rollouts: list[Rollout],
    *,
    n_calm: int = 650,
    n_instruct_mix: int = 500,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Calm transcripts (conversational) + Dolci-Instruct-SFT mix."""
    rng = random.Random(seed)
    calm = [{"messages": r.messages} for r in calm_rollouts]
    rng.shuffle(calm)
    calm = calm[:n_calm]

    mix = _load_dolci_instruct(n_instruct_mix, seed=seed)
    data = calm + mix
    rng.shuffle(data)
    return data


def _load_dolci_instruct(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Load `n` standard instruct samples from Dolci-Instruct-SFT.

    Falls back to an empty list (with a warning) if the dataset is unavailable;
    the SFT run then trains on calm data alone, which the paper notes is more
    prone to degeneration.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load Dolci-Instruct-SFT ({e}); "
              "SFT will use calm data only.")
        return []
