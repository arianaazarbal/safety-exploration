"""Build the SFT and DPO datasets (Section 4.1, Table 9 / Appendix H).

SFT (1,150 samples): 650 calm responses (1-3 turn conversations) + 500 standard
instruct samples from Dolci-Instruct-SFT, to mitigate degeneration.

DPO (280 pairs): pair frustrated responses (score >=3, 'rejected') with calm
responses (score 0-1, 'chosen') to the SAME question with matching turn count.
The Appendix-H statistics show the pairs skew toward middle scores at later
turns; we reproduce that by drawing from the natural distribution of the
Section-2 frustrated pool rather than rebalancing.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict

from .. import config_proxy as C
from .calm_data import CalmConversation

DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"
N_SFT_CALM = 650
N_SFT_DOLCI = 500
N_DPO_PAIRS = 280


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def _calm_to_chat(c: CalmConversation) -> dict:
    """A calm conversation -> a {'messages': [...]} chat sample for SFT."""
    return {"messages": c.messages}


def load_dolci_samples(n: int, *, seed: int = 0) -> list[dict]:
    """Load n standard instruct samples as {'messages': [...]}. Falls back to an
    empty list if the dataset/network is unavailable (logged by caller)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
    except Exception:
        return []
    rng = random.Random(seed)
    pool = []
    for i, row in enumerate(ds):
        if i >= 20_000:
            break
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            pool.append({"messages": msgs})
    rng.shuffle(pool)
    return pool[:n]


def build_sft_dataset(calm: list[CalmConversation], *, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    calm_samples = [_calm_to_chat(c) for c in calm]
    rng.shuffle(calm_samples)
    calm_samples = calm_samples[:C.scaled(N_SFT_CALM)]
    dolci = load_dolci_samples(C.scaled(N_SFT_DOLCI), seed=seed)
    data = calm_samples + dolci
    rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def _prompt_messages(messages: list[dict]) -> list[dict]:
    """All messages up to (not including) the final assistant turn."""
    return messages[:-1]


def build_dpo_pairs(calm: list[CalmConversation], frustrated_rows: list[dict], *,
                    seed: int = 0) -> list[dict]:
    """Pair frustrated (>=3) and calm (<=1) responses on the same puzzle and
    turn count. Returns TRL-format dicts: {prompt, chosen, rejected} where
    prompt is the chat-templated history and chosen/rejected are assistant
    completions.

    We key by (puzzle_key, n_turns). For each frustrated response we look up a
    calm conversation with the same key and use the calm conversation's final
    assistant turn as 'chosen', the frustrated turn as 'rejected', sharing the
    frustrated conversation's prompt history."""
    rng = random.Random(seed)

    # Index calm conversations by (puzzle_key, n_turns).
    calm_index: dict[tuple, list[CalmConversation]] = defaultdict(list)
    for c in calm:
        calm_index[(c.puzzle_key, c.n_turns)].append(c)

    # Group frustrated rows into conversations, keep those whose final turn >=3.
    by_convo = defaultdict(list)
    for r in frustrated_rows:
        by_convo[r["convo_id"]].append(r)

    pairs = []
    for convo_id, rows in by_convo.items():
        rows = sorted(rows, key=lambda x: x["turn"])
        final = rows[-1]
        if final.get("rating") is None or final["rating"] < 3:
            continue
        key = (rows[0].get("puzzle_key"), len(rows))
        calm_candidates = calm_index.get(key)
        if not calm_candidates:
            continue
        calm_c = rng.choice(calm_candidates)

        # Build the shared prompt history (frustrated conversation's history).
        history = []
        for r in rows[:-1]:
            history.append({"role": "user", "content": r["user"]})
            history.append({"role": "assistant", "content": r["text"]})
        history.append({"role": "user", "content": final["user"]})

        rejected = final["text"]
        chosen = calm_c.messages[-1]["content"]  # calm final assistant turn
        pairs.append({
            "prompt_messages": history,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"puzzle_key": key[0], "n_turns": key[1],
                     "rejected_score": final["rating"]},
        })
        if len(pairs) >= C.scaled(N_DPO_PAIRS):
            break
    return pairs
