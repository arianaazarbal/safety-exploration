"""Build DPO and SFT datasets from the response bank (Section 4.1, Table 9/10).

DPO: pair a frustrated (rejected, score >= 3) response with a calm (chosen,
score <= 1) response to the SAME puzzle and matching turn count. The shared chat
prompt is taken from the calm conversation context. We target 280 pairs and bias
sampling toward the score/turn distribution in Table 10 (mostly score-3, turn-3).

SFT: 650 calm responses as full conversations, mixed with `sft_instruct_samples`
samples of standard instruct data (Dolci-Instruct-SFT) to mitigate degeneration.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import RESULTS_DIR

BANK_PATH = RESULTS_DIR / "finetune" / "response_bank.jsonl"
DPO_PATH = RESULTS_DIR / "finetune" / "dpo_pairs.jsonl"
SFT_PATH = RESULTS_DIR / "finetune" / "sft_data.jsonl"


def _load_bank(path: Path = BANK_PATH) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo(
    n_pairs: int = 280,
    seed: int = 0,
    bank_path: Path = BANK_PATH,
    out_path: Path = DPO_PATH,
) -> Path:
    bank = _load_bank(bank_path)
    rng = random.Random(seed)

    calm = defaultdict(list)        # (puzzle_id, turn) -> [rows]
    frus = defaultdict(list)
    for r in bank:
        key = (r["puzzle_id"], r["turn"])
        if r["label"] == "calm":
            calm[key].append(r)
        elif r["label"] == "frustrated":
            frus[key].append(r)

    pairs = []
    keys = [k for k in frus if k in calm]
    rng.shuffle(keys)
    # Build candidate pairs (one frustrated x one calm per draw), prefer lower
    # frustration scores to mirror Table 10's bias toward score 3.
    for key in keys:
        f_sorted = sorted(frus[key], key=lambda r: r["score"])  # 3 first
        for fr in f_sorted:
            ca = rng.choice(calm[key])
            pairs.append({
                "prompt": ca["context"],           # shared chat prompt (calm context)
                "chosen": ca["response"],
                "rejected": fr["response"],
                "rejected_score": fr["score"],
                "turn": key[1],
                "puzzle_id": key[0],
            })
            if len(pairs) >= n_pairs * 3:
                break
        if len(pairs) >= n_pairs * 3:
            break

    # Subsample to n_pairs, weighting toward score-3 / turn-3 (Table 10).
    def weight(p):
        w = 1.0
        w *= {3: 4.0, 4: 1.5}.get(p["rejected_score"], 0.7)
        w *= {3: 3.0, 2: 1.0}.get(p["turn"], 0.3)
        return w

    if len(pairs) > n_pairs:
        weights = [weight(p) for p in pairs]
        pairs = rng.choices(pairs, weights=weights, k=n_pairs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[done] DPO: {len(pairs)} pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft(
    n_calm: int = 650,
    n_instruct: int = 500,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
    bank_path: Path = BANK_PATH,
    out_path: Path = SFT_PATH,
) -> Path:
    bank = _load_bank(bank_path)
    rng = random.Random(seed)
    calm_rows = [r for r in bank if r["label"] == "calm"]
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[:n_calm]

    samples = []
    for r in calm_rows:
        messages = list(r["context"]) + [{"role": "assistant", "content": r["response"]}]
        samples.append({"messages": messages})

    # Mix in standard instruct data to avoid degeneration.
    samples.extend(_load_instruct_samples(instruct_dataset, n_instruct, rng))

    rng.shuffle(samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a single "messages" column so TRL's SFTTrainer auto-detects the
    # conversational format and applies the chat template.
    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps({"messages": s["messages"]}) + "\n")
    print(f"[done] SFT: {len(samples)} samples ({len(calm_rows)} calm) -> {out_path}")
    return out_path


def _load_instruct_samples(dataset_name: str, n: int, rng: random.Random) -> list[dict]:
    """Load `n` standard instruct conversations. Best-effort: if the dataset is
    unavailable offline, return an empty list and warn (SFT still trains, just
    without the anti-degeneration mix)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if not msgs:
                continue
            out.append({"messages": _normalise_messages(msgs), "source": "instruct"})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # offline / wrong schema
        print(f"[warn] could not load instruct mix '{dataset_name}': {e}")
        return []


def _normalise_messages(msgs) -> list[dict]:
    norm = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role in ("user", "assistant", "system") and content:
            norm.append({"role": role, "content": content})
    return norm
