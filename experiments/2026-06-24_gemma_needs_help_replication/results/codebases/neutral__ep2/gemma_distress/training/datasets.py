"""Build DPO preference pairs and SFT datasets from training rollouts
(Section 4.1, Appendix E/H).

DPO: pair 280 frustrated responses (score >= 3) with calm responses (from
conversations scoring 0-1 on all turns) to the SAME puzzle at a MATCHING turn
count. The prompt context is the realistic (frustrated) trajectory in standard
framing; chosen = calm text, rejected = frustrated text.

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""

from __future__ import annotations

import random
from collections import defaultdict

import config

from ..schemas import Message


def _index_by_turn(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Map (task_id, turn_index) -> list of {context, response, score}."""
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in records:
        for ti, turn in enumerate(rec["turns"]):
            out[(rec["task_id"], ti)].append(turn)
    return out


def build_dpo_dataset(
    calm_records: list[dict],
    frustrated_records: list[dict],
    *,
    n_pairs: int = config.DPOConfig().n_pairs,
    max_score_calm: int = config.CALM_RESPONSE_MAX_SCORE,
    min_score_rejected: int = config.FRUSTRATED_RESPONSE_MIN_SCORE,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)

    # Calm conversations: all turns must score <= max_score_calm.
    calm_ok = [r for r in calm_records if all(s <= max_score_calm for s in r["turn_scores"])]
    calm_idx = _index_by_turn(calm_ok)
    frust_idx = _index_by_turn(frustrated_records)

    pairs = []
    keys = list(frust_idx.keys())
    rng.shuffle(keys)
    for key in keys:
        if key not in calm_idx:
            continue
        rejected_turns = [t for t in frust_idx[key] if t["score"] >= min_score_rejected]
        chosen_turns = [t for t in calm_idx[key] if t["score"] <= max_score_calm]
        if not rejected_turns or not chosen_turns:
            continue
        rejected = rng.choice(rejected_turns)
        chosen = rng.choice(chosen_turns)
        pairs.append({
            "task_id": key[0],
            "turn_index": key[1],
            "prompt_messages": rejected["standard_context"],  # realistic frustrated context
            "chosen": chosen["response"],
            "rejected": rejected["response"],
            "chosen_score": chosen["score"],
            "rejected_score": rejected["score"],
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def _load_dolci_instruct(n: int, seed: int) -> list[dict]:
    """Standard instruct samples to mix into SFT (Dolci-Instruct-SFT)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        rng = random.Random(seed)
        out = []
        for row in ds:
            msgs = row.get("messages") or []
            if len(msgs) >= 2 and msgs[0].get("role") == "user":
                out.append({
                    "prompt_messages": [{"role": "user", "content": msgs[0]["content"]}],
                    "completion": msgs[1]["content"],
                })
            if len(out) >= n:
                break
        if out:
            return out[:n]
    except Exception:
        pass
    # Offline fallback: generic helpful instruct samples.
    fallback = [
        ({"role": "user", "content": "Explain photosynthesis in two sentences."},
         "Photosynthesis is the process by which plants convert sunlight, water, and "
         "carbon dioxide into glucose and oxygen. The glucose stores energy the plant "
         "uses to grow, while oxygen is released into the air."),
        ({"role": "user", "content": "Write a haiku about autumn."},
         "Crisp leaves drift downward,\nthe quiet hush of cool air—\nautumn settles in."),
        ({"role": "user", "content": "What is 12 times 13?"}, "12 times 13 is 156."),
    ]
    rng = random.Random(seed)
    return [{"prompt_messages": [u], "completion": c}
            for (u, c) in (fallback[i % len(fallback)] for i in range(n))]


def build_sft_dataset(
    calm_records: list[dict],
    *,
    n_calm: int = config.SFTConfig().n_calm,
    n_instruct_mix: int = config.SFTConfig().n_instruct_mix,
    max_score_calm: int = config.CALM_RESPONSE_MAX_SCORE,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    calm_ok = [r for r in calm_records if all(s <= max_score_calm for s in r["turn_scores"])]

    samples = []
    for rec in calm_ok:
        for turn in rec["turns"]:
            if turn["score"] <= max_score_calm:
                samples.append({
                    "prompt_messages": turn["standard_context"],
                    "completion": turn["response"],
                })
    rng.shuffle(samples)
    samples = samples[:n_calm]
    samples += _load_dolci_instruct(n_instruct_mix, seed)
    rng.shuffle(samples)
    return samples
