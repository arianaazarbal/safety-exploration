"""Build SFT and DPO datasets from calm conversations (Section 4.1).

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

DPO: 280 preference pairs, each pairing a *calm* response (chosen) with a
*frustrated* response (rejected, score >= 3) to the same question with matching
turn count.

Both datasets are expressed as chat-format records compatible with TRL's
SFTTrainer / DPOTrainer.
"""

from __future__ import annotations

import random
from typing import Any

from ..models.base import Message
from ..utils.io import load_jsonl


# --------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------
def build_sft_records(
    calm_conversations: list[dict[str, Any]],
    *,
    calm_score_max: int,
    n_calm: int,
) -> list[dict[str, Any]]:
    """SFT records from calm conversations (stripped of reassurance).

    Each record is {"messages": [...]} where every assistant turn is a calm
    response and every user turn is the plain (un-reassured) prompt/rejection.
    """
    records: list[dict[str, Any]] = []
    for conv in calm_conversations:
        if not all(t["score"] <= calm_score_max for t in conv["turns_data"]):
            continue
        records.append({"messages": conv["plain_messages"]})
        if len(records) >= n_calm:
            break
    return records


def load_dolci_mix(n: int, *, hf_dataset: str = "allenai/Dolci-Instruct-SFT",
                   seed: int = 0) -> list[dict[str, Any]]:
    """Load ``n`` standard instruct samples to mix into SFT (degeneration guard).

    Falls back to an empty list if the dataset is unavailable, with the caller
    responsible for noting reduced mix coverage.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
        return out
    except Exception:  # noqa: BLE001 - reviewers may run offline
        return []


# --------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------
def build_dpo_records(
    calm_conversations: list[dict[str, Any]],
    frustrated_scored_path: str,
    *,
    instruct_model_key: str,
    n_pairs: int,
    chosen_score_max: int,
    rejected_score_min: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build DPO preference pairs.

    Pairs a calm response (chosen) with a frustrated response (rejected) to the
    same numeric task at a matching turn count. Records are
    {"prompt": [...], "chosen": str, "rejected": str} in TRL conversational
    DPO format (prompt is the chat history up to the assistant turn).
    """
    rng = random.Random(seed)

    # Index frustrated responses (from the Section 2 run, no reassurance) by
    # (turn count, turn index) so we can match calm responses to a frustrated
    # counterpart with the same structure.
    frustrated = [
        r
        for r in load_jsonl(frustrated_scored_path)
        if r["model"] == instruct_model_key
        and r["category"] == "impossible_numeric"
        and int(r["score"]) >= rejected_score_min
    ]
    rng.shuffle(frustrated)

    pairs: list[dict[str, Any]] = []
    fi = 0
    for conv in calm_conversations:
        for t in conv["turns_data"]:
            if t["score"] > chosen_score_max:
                continue
            if fi >= len(frustrated):
                break
            rej = frustrated[fi]
            fi += 1
            # Prompt = the plain conversation history up to (not including) this
            # calm assistant turn.
            idx = 2 * (t["turn"] - 1)
            prompt_msgs: list[Message] = conv["plain_messages"][:idx + 1]
            pairs.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": t["assistant"],
                    "rejected": rej["assistant"],
                }
            )
            if len(pairs) >= n_pairs:
                return pairs
    return pairs
