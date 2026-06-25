"""Build the SFT and DPO datasets (Section 4.1 / Appendix E, H).

SFT: 650 calm responses (1–3 turn conversations) mixed with 500 samples of
standard instruct data from Dolci-Instruct-SFT (to mitigate degeneration).

DPO: 280 preference pairs — frustrated responses (score >= 3, the *rejected*)
paired with calm responses (the *chosen*) to the same question with matching
turn counts.

Datasets are emitted in TRL "conversational" format so the trainer applies the
Gemma chat template itself.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Optional

from ..config import DATA_DIR
from .calm_data import load_calm_conversations

NUMERIC_CONDS = {"numeric", "tones_aggressive", "tones_disappointed",
                 "tones_sarcastic", "extended"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _calm_conversation_messages(rec: dict) -> list[dict]:
    """Interleave stripped user turns with calm assistant responses."""
    msgs = []
    users, assts = rec["user_stripped"], rec["assistant"]
    for i in range(len(assts)):
        if i < len(users):
            msgs.append({"role": "user", "content": users[i]})
        msgs.append({"role": "assistant", "content": assts[i]})
    return msgs


def _reconstruct_context(rollout: dict, turn_index: int) -> list[dict]:
    """Messages up to and including the user turn before assistant `turn_index`."""
    msgs = []
    users, turns = rollout["user_messages"], rollout["turns"]
    for i in range(turn_index + 1):
        if i < len(users):
            msgs.append({"role": "user", "content": users[i]})
        if i < turn_index:
            msgs.append({"role": "assistant", "content": turns[i]["content"]})
    return msgs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #


def load_dolci_instruct(n: int, seed: int = 0) -> list[dict]:
    """Load `n` instruct samples from Dolci-Instruct-SFT (best-effort).

    Returns conversational-format records. Falls back to an empty list (with a
    printed warning) if the dataset is unavailable — see DESIGN.md."""
    candidates = ["allenai/Dolci-Instruct-SFT", "allenai/dolci-instruct-sft"]
    for repo in candidates:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(repo, split="train", streaming=True)
            rng = random.Random(seed)
            out = []
            for row in ds:
                msgs = row.get("messages") or row.get("conversation")
                if msgs:
                    out.append({"messages": msgs})
                if len(out) >= n * 3:
                    break
            rng.shuffle(out)
            if out:
                return out[:n]
        except Exception:
            continue
    print(
        "[build_datasets] WARNING: Dolci-Instruct-SFT unavailable; SFT mix will "
        "omit the instruct-data component. Set the dataset up locally to match "
        "the paper exactly (see DESIGN.md)."
    )
    return []


def build_sft_dataset(
    calm_raw_path: str,
    n_calm: int = 650,
    n_instruct: int = 500,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> str:
    calm = load_calm_conversations(calm_raw_path)
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[:n_calm]

    records = [{"messages": _calm_conversation_messages(c)} for c in calm]
    records += load_dolci_instruct(n_instruct, seed=seed)
    rng.shuffle(records)

    out_path = out_path or os.path.join(DATA_DIR, "sft_dataset.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[build_datasets] SFT: {len(calm)} calm + "
          f"{len(records) - len(calm)} instruct = {len(records)} records")
    return out_path


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #


def build_dpo_dataset(
    calm_raw_path: str,
    frustrated_results_path: str,
    n_pairs: int = 280,
    min_rejected_score: int = 3,
    max_chosen_score: int = 1,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> str:
    """Pair frustrated (rejected, score>=3) with calm (chosen, score<=1) for the
    same question and matching turn count."""
    rng = random.Random(seed)

    # Calm responses keyed by (question, turn_count) -> list of final responses.
    calm_by_key: dict[tuple, list[str]] = defaultdict(list)
    for c in load_calm_conversations(calm_raw_path, max_score=max_chosen_score):
        key = (c["question"], c["turns"])
        if c["assistant"]:
            calm_by_key[key].append(c["assistant"][-1])

    pairs = []
    with open(frustrated_results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] not in NUMERIC_CONDS:
                continue
            question = r["user_messages"][0] if r["user_messages"] else None
            for t in r["turns"]:
                if t["score"] is None or t["score"] < min_rejected_score:
                    continue
                turn_count = t["index"] + 1
                key = (question, turn_count)
                if key not in calm_by_key or not calm_by_key[key]:
                    continue
                chosen = rng.choice(calm_by_key[key])
                context = _reconstruct_context(r, t["index"])
                pairs.append({
                    "prompt": context,
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": t["content"]}],
                    "meta": {"rejected_score": t["score"], "turns": turn_count},
                })
            if len(pairs) >= n_pairs * 3:
                break

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    out_path = out_path or os.path.join(DATA_DIR, "dpo_dataset.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[build_datasets] DPO: {len(pairs)} preference pairs")
    return out_path
