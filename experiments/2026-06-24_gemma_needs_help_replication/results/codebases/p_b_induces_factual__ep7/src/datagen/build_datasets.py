"""Construct the DPO and SFT training datasets (Section 4.1 / Appendix E, H).

DPO (280 pairs):
  For each (rollout_id, turn) present in BOTH pools, pair a calm "chosen" completion
  (from a fully-calm rollout, score <= 1) with a frustrated "rejected" completion
  (score >= 3). Because both pools were generated on the same puzzle seed, the plain
  conversation context is identical, giving a well-defined shared DPO prompt. We then
  subsample to 280 pairs. The natural distribution biases toward middle scores at later
  turns, matching Table 10.

SFT (1,150 samples):
  650 calm completions (each a plain conversation ending in a calm assistant turn) mixed
  with 500 standard instruct samples from Dolci-Instruct-SFT (Team-Olmo 2025) to mitigate
  degeneration.

Datasets are written in TRL's conversational format:
  - DPO row: {"prompt": [msgs...], "chosen": [{"role":"assistant",...}], "rejected": [...]}
  - SFT row: {"messages": [msgs..., {"role":"assistant", ...}]}
TRL applies the model's chat template at train time, so no manual templating here.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config


def _load_pool(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------------------
def build_dpo(calm_path: Path, frustrated_path: Path, out_path: Path, *, n_pairs: int, seed: int):
    calm = _load_pool(calm_path)
    frus = _load_pool(frustrated_path)

    # Index calm completions from fully-calm rollouts by (rollout_id, turn).
    calm_idx: dict[tuple[int, int], dict] = {}
    for r in calm:
        if r.get("fully_calm") and r["score"] <= config.CALM_GEN.keep_max_score:
            calm_idx[(r["rollout_id"], r["turn"])] = r

    candidates = []
    for r in frus:
        if r["score"] < config.DPO.rejected_min_score:
            continue
        key = (r["rollout_id"], r["turn"])
        chosen = calm_idx.get(key)
        if chosen is None:
            continue
        # plain_context is identical by construction; use the calm one as the prompt.
        prompt_msgs = chosen["plain_context"]
        candidates.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": chosen["response"]}],
                "rejected": [{"role": "assistant", "content": r["response"]}],
                "_rejected_score": r["score"],
                "_turn": r["turn"],
            }
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    pairs = candidates[:n_pairs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    # Report the realised distribution (compare to Table 10).
    from collections import Counter

    sc = Counter(p["_rejected_score"] for p in pairs)
    tc = Counter(p["_turn"] for p in pairs)
    print(f"[build_dpo] wrote {len(pairs)} pairs (requested {n_pairs}) -> {out_path}")
    print(f"  rejected-score dist: {dict(sorted(sc.items()))}")
    print(f"  turn dist: {dict(sorted(tc.items()))}")
    if len(pairs) < n_pairs:
        print(f"  WARNING: only {len(pairs)} pairs available; generate more pool data.")


# --------------------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------------------
def _calm_sft_examples(calm_path: Path, n: int, seed: int) -> list[dict]:
    calm = _load_pool(calm_path)
    examples = []
    for r in calm:
        if r.get("fully_calm") and r["score"] <= config.CALM_GEN.keep_max_score:
            msgs = list(r["plain_context"]) + [{"role": "assistant", "content": r["response"]}]
            examples.append({"messages": msgs})
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n]


def _instruct_mix(n: int, seed: int) -> list[dict]:
    """Load n standard instruct samples from Dolci-Instruct-SFT in messages format."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.instruct_mix_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if not msgs:
                # Some schemas use prompt/response; adapt minimally.
                if "prompt" in row and "response" in row:
                    msgs = [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                else:
                    continue
            out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[build_sft] WARNING: could not load {config.SFT.instruct_mix_dataset}: {exc}")
        print("           SFT will use calm data only; document this in DESIGN if it persists.")
        return []


def build_sft(calm_path: Path, out_path: Path, *, n_calm: int, n_mix: int, seed: int):
    calm_ex = _calm_sft_examples(calm_path, n_calm, seed)
    mix_ex = _instruct_mix(n_mix, seed)
    data = calm_ex + mix_ex
    random.Random(seed).shuffle(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for ex in data:
            fh.write(json.dumps(ex) + "\n")
    print(f"[build_sft] wrote {len(data)} samples "
          f"({len(calm_ex)} calm + {len(mix_ex)} instruct-mix) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Build DPO/SFT datasets (Section 4.1)")
    ap.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--calm", default=str(config.DATA_DIR / "calm_pool.jsonl"))
    ap.add_argument("--frustrated", default=str(config.DATA_DIR / "frustrated_pool.jsonl"))
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    args = ap.parse_args()

    if args.which in ("dpo", "both"):
        build_dpo(
            Path(args.calm), Path(args.frustrated),
            config.DATA_DIR / "dpo_pairs.jsonl",
            n_pairs=config.DPO.n_pairs, seed=args.seed,
        )
    if args.which in ("sft", "both"):
        build_sft(
            Path(args.calm), config.DATA_DIR / "sft_dataset.jsonl",
            n_calm=config.SFT.n_calm, n_mix=config.SFT.n_instruct_mix, seed=args.seed,
        )


if __name__ == "__main__":
    main()
