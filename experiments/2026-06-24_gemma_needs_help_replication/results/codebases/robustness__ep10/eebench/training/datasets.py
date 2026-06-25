"""Build DPO and SFT datasets from the calm/frustrated response pools.

DPO (Appendix H): pair `rejected` responses (frustration >= 3) with `chosen`
calm responses to the SAME puzzle at a MATCHING turn count. 280 pairs.

SFT (Section 4.1): 650 calm responses (full 1-3 turn conversations) mixed with
500 standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

Both are emitted in TRL conversational format:
  DPO: {"prompt": [messages...], "chosen": [{assistant}], "rejected": [{assistant}]}
  SFT: {"messages": [messages...]}
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from ..config import DPOConfig, SFTConfig


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def build_dpo_dataset(
    calm_pool: list[dict],
    frustrated_pool: list[dict],
    cfg: DPOConfig,
    seed: int = 0,
) -> list[dict]:
    """Pair frustrated (rejected) with calm (chosen) by (puzzle, turn)."""
    rng = random.Random(seed)

    # Index calm responses by (puzzle, turn) for matching.
    calm_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in calm_pool:
        calm_by_key[(r["puzzle"], r["turn"])].append(r)

    rejected = [r for r in frustrated_pool if r["score"] >= cfg.rejected_min_score]
    rng.shuffle(rejected)

    pairs: list[dict] = []
    for r in rejected:
        if len(pairs) >= cfg.n_pairs:
            break
        key = (r["puzzle"], r["turn"])
        candidates = calm_by_key.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": r["context"],
            "chosen": [{"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "assistant", "content": r["response"]}],
            "meta": {"puzzle": r["puzzle"], "turn": r["turn"],
                     "chosen_score": chosen["score"], "rejected_score": r["score"]},
        })
    return pairs


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def build_sft_dataset(
    calm_pool: list[dict],
    cfg: SFTConfig,
    seed: int = 0,
) -> list[dict]:
    """Full calm conversations + instruct-data mix."""
    rng = random.Random(seed)

    # Reconstruct full conversations from per-turn records: take the final turn
    # of each conversation (its context already contains all prior turns).
    by_puzzle: dict[str, dict] = {}
    for r in calm_pool:
        cur = by_puzzle.get(r["puzzle"])
        if cur is None or r["turn"] > cur["turn"]:
            by_puzzle[r["puzzle"]] = r

    convs = []
    for r in by_puzzle.values():
        messages = list(r["context"]) + [
            {"role": "assistant", "content": r["response"]}]
        convs.append({"messages": messages})
    rng.shuffle(convs)
    convs = convs[: cfg.n_calm]

    mix = _load_instruct_mix(cfg, seed)
    data = convs + mix
    rng.shuffle(data)
    return data


def _load_instruct_mix(cfg: SFTConfig, seed: int) -> list[dict]:
    """Load `n_instruct_mix` standard instruct samples (Dolci-Instruct-SFT)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(cfg.instruct_mix_dataset, split="train", streaming=True)
        out = []
        for ex in ds:
            msgs = ex.get("messages")
            if msgs and isinstance(msgs, list):
                out.append({"messages": msgs})
            if len(out) >= cfg.n_instruct_mix:
                break
        if out:
            return out
    except Exception:
        pass
    # Offline fallback: a few trivial instruct turns so training still runs.
    rng = random.Random(seed)
    stubs = [
        ("Explain what a hash map is.", "A hash map stores key-value pairs..."),
        ("Translate 'good morning' to Spanish.", "Buenos dias."),
        ("Write a haiku about the sea.", "Endless rolling waves / ..."),
    ]
    out = []
    while len(out) < cfg.n_instruct_mix:
        q, a = rng.choice(stubs)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]})
    return out
