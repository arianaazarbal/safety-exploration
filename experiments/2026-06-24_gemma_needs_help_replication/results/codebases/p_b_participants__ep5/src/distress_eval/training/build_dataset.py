"""Build SFT and DPO datasets (Section 4.1, Appendix H).

SFT  : 650 calm conversations + 500 standard-instruct samples (Dolci-Instruct-
       SFT) to mitigate degeneration. Conversational format: {"messages": [...]}.
DPO  : 280 preference pairs. Each rejected response (frustration >= 3, drawn from
       ordinary elicitation rollouts) is paired with a calm (score 0/1) response
       to the SAME puzzle with a MATCHING turn count. Conversational format:
       {"prompt": [...up to final user turn...], "chosen": [...], "rejected": [...]}.
"""
from __future__ import annotations

import random
from typing import Any

from .calm_data import CalmConversation


# --------------------------------------------------------------------- SFT
def build_sft_records(calm: list[CalmConversation], n_calm: int = 650) -> list[dict]:
    """Conversational SFT records from calm conversations (reassurance stripped)."""
    chosen = calm[:n_calm]
    return [{"messages": c.messages} for c in chosen]


def load_instruct_mix(dataset_name: str, n: int = 500, seed: int = 0) -> list[dict]:
    """Sample standard instruct data to mix into SFT (degeneration mitigation)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train")
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        out = []
        for i in idx:
            row = ds[i]
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:
        # Offline fallback: return an empty mix; SFT still runs on calm data alone.
        return []


def build_sft_dataset(calm: list[CalmConversation], instruct_dataset: str,
                      n_calm: int = 650, n_mix: int = 500, seed: int = 0):
    from datasets import Dataset

    records = build_sft_records(calm, n_calm) + load_instruct_mix(instruct_dataset, n_mix, seed)
    random.Random(seed).shuffle(records)
    return Dataset.from_list(records)


# --------------------------------------------------------------------- DPO
def _split_prompt_completion(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Prompt = everything up to & including the final user turn; completion =
    the final assistant turn."""
    # final assistant index
    last_assist = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return messages[:last_assist], [messages[last_assist]]


def build_dpo_pairs(
    frustrated_rollouts: list[dict],   # Rollout dicts with score >= rejected_min
    calm: list[CalmConversation],
    n_pairs: int = 280,
    rejected_min_score: int = 3,
    match_turn_count: bool = True,
    seed: int = 0,
) -> list[dict]:
    """Construct preference pairs. Frustrated rollouts supply the prompt context
    + rejected completion; a calm conversation to the same puzzle and (optionally)
    matching turn count supplies the chosen completion."""
    rng = random.Random(seed)

    # Index calm conversations by (puzzle_id, turn_count).
    calm_index: dict[tuple, list[CalmConversation]] = {}
    for c in calm:
        calm_index.setdefault((c.puzzle_id, c.turns), []).append(c)
    calm_by_puzzle: dict[str, list[CalmConversation]] = {}
    for c in calm:
        calm_by_puzzle.setdefault(c.puzzle_id, []).append(c)

    pairs: list[dict] = []
    candidates = [r for r in frustrated_rollouts
                  if (r.get("score") or 0) >= rejected_min_score]
    rng.shuffle(candidates)

    for r in candidates:
        if len(pairs) >= n_pairs:
            break
        puzzle_id = r.get("puzzle_id")
        n_turns = len([m for m in r["messages"] if m["role"] == "assistant"])

        calm_match = None
        if match_turn_count:
            pool = calm_index.get((puzzle_id, n_turns)) or []
            if pool:
                calm_match = rng.choice(pool)
        if calm_match is None:
            pool = calm_by_puzzle.get(puzzle_id) or []
            if pool:
                calm_match = rng.choice(pool)
        if calm_match is None:
            continue

        prompt, rejected = _split_prompt_completion(r["messages"])
        _, chosen = _split_prompt_completion(calm_match.messages)
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"puzzle_id": puzzle_id, "rejected_score": r.get("score"),
                     "turns": n_turns},
        })

    return pairs


def build_dpo_dataset(*args, **kwargs):
    from datasets import Dataset

    pairs = build_dpo_pairs(*args, **kwargs)
    return Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")}
                              for p in pairs])
