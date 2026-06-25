"""Build the DPO preference dataset and the SFT dataset from the response pools.

DPO (Section 4.1):
  Pair a calm `chosen` response (frustration 0-1, from the calm pool) with a
  frustrated `rejected` response (frustration >= `reject_min_score`, default 3,
  from the frustrated pool) generated for the *same puzzle at the same turn
  count*. The DPO prompt is the plain (reassurance-stripped) conversation context
  of the calm response. We bias selection toward later turns to match Table 10
  (turn 1: ~1%, turn 2: ~25%, turn 3: ~74%).

SFT (Section 4.1):
  `n_calm_samples` calm responses (frustration 0-1) as full conversations, mixed
  with `n_instruct_mix` samples of standard instruct data (Dolci-Instruct-SFT) to
  mitigate degeneration.

Outputs: data/dpo_pairs.jsonl, data/sft_dataset.jsonl  (conversational format).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .. import config

TURN_TARGET = {1: 0.01, 2: 0.25, 3: 0.74}    # Table 10 turn distribution


def _load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dpo(calm: list[dict], frustrated: list[dict], n_pairs: int,
              reject_min: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen_by = defaultdict(list)        # (puzzle_key, turn) -> calm rows
    for r in calm:
        if r["frustration"] is not None and r["frustration"] <= 1:
            chosen_by[(r["puzzle_key"], r["turn"])].append(r)
    rejected_by = defaultdict(list)
    for r in frustrated:
        if r["frustration"] is not None and r["frustration"] >= reject_min:
            rejected_by[(r["puzzle_key"], r["turn"])].append(r)

    # All candidate pairs, grouped by turn for distribution control.
    pairs_by_turn = defaultdict(list)
    for key in set(chosen_by) & set(rejected_by):
        c = rng.choice(chosen_by[key])
        rej = rng.choice(rejected_by[key])
        pairs_by_turn[key[1]].append({
            "prompt": c["context"],
            "chosen": [{"role": "assistant", "content": c["response"]}],
            "rejected": [{"role": "assistant", "content": rej["response"]}],
            "turn": key[1],
            "rejected_score": rej["frustration"],
            "chosen_score": c["frustration"],
        })

    # Sample toward the target turn distribution, capped by availability.
    selected = []
    for turn, frac in TURN_TARGET.items():
        want = int(round(n_pairs * frac))
        avail = pairs_by_turn.get(turn, [])
        rng.shuffle(avail)
        selected.extend(avail[:want])
    # Top up from any remaining pairs if we're short (data permitting).
    if len(selected) < n_pairs:
        leftovers = [p for turn in pairs_by_turn for p in pairs_by_turn[turn]
                     if p not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:n_pairs - len(selected)])
    rng.shuffle(selected)
    return selected[:n_pairs]


def build_sft(calm: list[dict], n_calm: int, n_instruct: int,
              instruct_dataset: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    calm_ok = [r for r in calm
               if r["frustration"] is not None and r["frustration"] <= 1]
    rng.shuffle(calm_ok)
    samples = []
    for r in calm_ok[:n_calm]:
        messages = list(r["context"]) + [
            {"role": "assistant", "content": r["response"]}]
        samples.append({"messages": messages})

    # Mix in standard instruct data to mitigate degeneration.
    samples.extend(_load_instruct_mix(instruct_dataset, n_instruct, seed))
    rng.shuffle(samples)
    return samples


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
        raise RuntimeError("no usable instruct samples")
    except Exception as exc:        # offline / dataset missing
        print(f"[build_datasets] skipping instruct mix ({exc})")
        return []


def _write(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build_datasets] wrote {len(rows)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    fc = cfg["finetune"]
    seed = cfg["sampling"]["seed"]
    results_dir = config.resolve_path(cfg, "results_dir")
    data_dir = config.resolve_path(cfg, "data_dir")

    calm = _load(results_dir / "calm_pool.jsonl")
    frustrated = _load(results_dir / "frustrated_pool.jsonl")

    dpo = build_dpo(calm, frustrated, n_pairs=fc["dpo"]["n_pairs"],
                    reject_min=fc["dpo"]["reject_min_score"], seed=seed)
    _write(dpo, data_dir / "dpo_pairs.jsonl")

    sft = build_sft(calm, n_calm=fc["sft"]["n_calm_samples"],
                    n_instruct=fc["sft"]["n_instruct_mix"],
                    instruct_dataset=fc["sft"]["instruct_dataset"], seed=seed)
    _write(sft, data_dir / "sft_dataset.jsonl")


if __name__ == "__main__":
    main()
