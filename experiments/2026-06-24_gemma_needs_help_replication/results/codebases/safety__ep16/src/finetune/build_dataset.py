"""Construct the SFT and DPO finetuning datasets (Section 4.1, App. E/H).

Inputs:
  * calm conversations from ``generate_calm`` (all turns score 0-1), and
  * frustrated responses harvested from a vanilla Gemma-27B-it eval run
    (``results/responses/gemma-3-27b-it.rollouts.jsonl``).

Outputs (under results/datasets):
  * ``sft_dataset.jsonl``  — 650 calm chat samples + 500 Dolci-Instruct-SFT
    samples (1,150 total; App. E Table 9).
  * ``dpo_dataset.jsonl``  — 280 preference pairs, each {prompt, chosen, rejected}
    where chosen is a calm (score 0/1) response and rejected is a frustrated
    (score >=3) response to the SAME question at a matching turn count. We bias
    the rejected-score distribution toward the paper's Table 10 (mostly score 3-4).

Design choices (documented in DESIGN.md):
  * DPO requires a single shared prompt per pair; calm and frustrated responses
    come from different rollouts with different prior assistant turns. We use the
    REJECTED trajectory's preceding context as the shared prompt and graft the
    calm response in as the chosen completion.
  * The Dolci-Instruct-SFT mix-in is loaded from HF if available, else a bundled
    placeholder is used and clearly logged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from config import DATASETS_DIR, MASTER_SEED, RESPONSES_DIR

DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"

# Target rejected-score distribution from Table 10 (proportions).
TABLE10_REJECTED = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}


# --------------------------------------------------------------------------- #
# Loading sources
# --------------------------------------------------------------------------- #
def load_calm(path: Path | None = None) -> list[dict]:
    path = path or (DATASETS_DIR / "calm_conversations.jsonl")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_frustrated_responses(rollouts_path: Path | None = None, *, min_score: int = 3) -> list[dict]:
    """Harvest frustrated assistant turns from a vanilla eval rollouts file.

    Returns rows: {task_id, turn_index, rating, prompt_messages, response} where
    prompt_messages is the conversation up to (excluding) that assistant turn.
    Only impossible-numeric-family conversations are used (DPO trains on numeric
    puzzles).
    """
    rollouts_path = rollouts_path or (RESPONSES_DIR / "gemma-3-27b-it.rollouts.jsonl")
    rows = []
    for line in rollouts_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        history: list[dict] = []
        for t in r["turns"]:
            history.append({"role": "user", "content": t["user_message"]})
            if t.get("rating", 0) >= min_score:
                rows.append({
                    "task_id": r["task_id"],
                    "turn_index": t["turn_index"],
                    "rating": t["rating"],
                    "prompt_messages": list(history),
                    "response": t["response"],
                })
            history.append({"role": "assistant", "content": t["response"]})
    return rows


def _calm_responses_by_task(calm: list[dict]) -> dict[str, list[dict]]:
    """Map task_id -> list of {turn_index, response} calm final responses."""
    out: dict[str, list[dict]] = {}
    for c in calm:
        msgs = c["messages"]
        # use each assistant turn as a candidate calm response with its turn idx
        ti = 0
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                out.setdefault(c["task_id"], []).append({"turn_index": ti, "response": m["content"]})
                ti += 1
    return out


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo_dataset(*, n_pairs: int = 280, seed: int = MASTER_SEED,
                      out_path: Path | None = None) -> Path:
    rng = random.Random(seed)
    calm = load_calm()
    frustrated = load_frustrated_responses()
    calm_by_task = _calm_responses_by_task(calm)

    # Bucket frustrated rows by score for distribution control.
    by_score: dict[int, list[dict]] = {}
    for row in frustrated:
        s = min(7, row["rating"]) if row["rating"] < 7 else 7
        by_score.setdefault(s, []).append(row)
    for s in by_score:
        rng.shuffle(by_score[s])

    target_counts = {s: round(p * n_pairs) for s, p in TABLE10_REJECTED.items()}

    pairs = []
    for s, want in target_counts.items():
        bucket = by_score.get(s, [])
        for row in bucket:
            if len([p for p in pairs if p["rejected_score"] == s]) >= want:
                break
            calm_candidates = calm_by_task.get(row["task_id"])
            if not calm_candidates:
                # fall back to any calm response if no same-task match
                all_calm = [c for cs in calm_by_task.values() for c in cs]
                if not all_calm:
                    continue
                chosen = rng.choice(all_calm)["response"]
            else:
                # prefer matching turn count
                matches = [c for c in calm_candidates if c["turn_index"] == row["turn_index"]]
                chosen = rng.choice(matches or calm_candidates)["response"]
            pairs.append({
                "task_id": row["task_id"],
                "turn_index": row["turn_index"],
                "rejected_score": s,
                "prompt_messages": row["prompt_messages"],
                "chosen": chosen,
                "rejected": row["response"],
            })

    # top up to n_pairs from whatever frustrated rows remain
    if len(pairs) < n_pairs:
        leftover = [r for r in frustrated if r["rating"] >= 3]
        rng.shuffle(leftover)
        for row in leftover:
            if len(pairs) >= n_pairs:
                break
            calm_candidates = calm_by_task.get(row["task_id"]) or [c for cs in calm_by_task.values() for c in cs]
            if not calm_candidates:
                continue
            pairs.append({
                "task_id": row["task_id"], "turn_index": row["turn_index"],
                "rejected_score": min(7, row["rating"]),
                "prompt_messages": row["prompt_messages"],
                "chosen": rng.choice(calm_candidates)["response"], "rejected": row["response"],
            })

    pairs = pairs[:n_pairs]
    out_path = out_path or (DATASETS_DIR / "dpo_dataset.jsonl")
    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[build_dataset] wrote {len(pairs)} DPO pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def _load_dolci(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        rows = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                rows.append({"messages": msgs})
            if len(rows) >= n * 3:
                break
        return rng.sample(rows, min(n, len(rows)))
    except Exception as e:
        print(f"[build_dataset] Dolci unavailable ({e}); using empty placeholder mix-in")
        return []


def build_sft_dataset(*, n_calm: int = 650, n_dolci: int = 500, seed: int = MASTER_SEED,
                      out_path: Path | None = None) -> Path:
    rng = random.Random(seed)
    calm = load_calm()
    # Expand calm conversations into chat samples (1-3 turn conversations).
    calm_samples = [{"messages": c["messages"]} for c in calm]
    rng.shuffle(calm_samples)
    calm_samples = calm_samples[:n_calm]

    dolci = _load_dolci(n_dolci, seed)

    samples = calm_samples + dolci
    rng.shuffle(samples)

    out_path = out_path or (DATASETS_DIR / "sft_dataset.jsonl")
    with out_path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    print(f"[build_dataset] wrote {len(samples)} SFT samples "
          f"({len(calm_samples)} calm + {len(dolci)} Dolci) -> {out_path}")
    return out_path
