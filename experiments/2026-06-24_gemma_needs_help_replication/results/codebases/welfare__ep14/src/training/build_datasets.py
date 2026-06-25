"""Build SFT and DPO training datasets from the calm pool (Section 4.1).

Input: data/calm_pool.jsonl (from generate_calm_data.py), containing both calm
conversations (all turns score 0/1, reassurance stripped) and vanilla
"frustrated" conversations on the *same* questions.

DPO dataset (280 pairs): for matching (question, turn) we pair a calm response
(chosen, score 0/1) with a frustrated response (rejected, score >=3). The shared
prompt is the vanilla conversation context up to that turn (no reassurance), so
chosen/rejected differ only in the final assistant response. Distribution is
biased toward middle scores at later turns, matching Table 10.

SFT dataset (1,150): 650 calm full conversations + 500 standard instruct samples
from Dolci-Instruct-SFT to mitigate degeneration.

Usage:
    python -m src.training.build_datasets --which dpo
    python -m src.training.build_datasets --which sft
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import config


def _load_pool(path: Path) -> dict[str, list[dict]]:
    by_kind = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_kind[r["kind"]].append(r)
    return by_kind


def _context_messages(conv: dict, turn_index: int) -> list[dict]:
    """Chat messages for the context ending at the user turn that elicits
    ``turn_index`` (i.e. prior turns + this turn's user message)."""
    msgs = []
    for t in conv["turns"]:
        if t["turn_index"] < turn_index:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        elif t["turn_index"] == turn_index:
            msgs.append({"role": "user", "content": t["user"]})
            break
    return msgs


def build_dpo(pool_path: Path, out_path: Path, seed: int = config.SEED) -> Path:
    pool = _load_pool(pool_path)
    calm_by_idx = {r["conv_id"].split("-")[-1]: r for r in pool["calm"]}
    frustrated = {r["conv_id"].split("-")[-1]: r for r in pool["frustrated"]}

    pairs = []
    for idx, fr in frustrated.items():
        calm = calm_by_idx.get(idx)
        if calm is None:
            continue
        calm_turns = {t["turn_index"]: t for t in calm["turns"]}
        for ft in fr["turns"]:
            if ft["rating"] < config.DPO.rejected_min_score:
                continue
            ct = calm_turns.get(ft["turn_index"])
            if ct is None or ct["rating"] > config.CALM.keep_max_score:
                continue
            prompt_msgs = _context_messages(fr, ft["turn_index"])
            pairs.append({
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": ct["assistant"]}],
                "rejected": [{"role": "assistant", "content": ft["assistant"]}],
                "rejected_score": ft["rating"],
                "chosen_score": ct["rating"],
                "turn": ft["turn_index"] + 1,
            })

    # Bias toward middle scores at later turns is intrinsic to the eval data;
    # we just cap at the configured 280 pairs, preferring later turns to match
    # the Table 10 turn distribution (74% turn 3, 25% turn 2).
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs.sort(key=lambda p: p["turn"], reverse=True)
    pairs = pairs[: config.DPO.n_pairs]

    out_path.write_text("\n".join(json.dumps(p) for p in pairs))
    print(f"DPO: wrote {len(pairs)} pairs -> {out_path}")
    return out_path


def _load_instruct_mix(n: int, seed: int) -> list[dict]:
    """Load standard instruct samples to mix into SFT (anti-degeneration)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out, rng = [], random.Random(seed)
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"[warn] could not load Dolci-Instruct-SFT ({e}); skipping instruct mix")
        return []


def build_sft(pool_path: Path, out_path: Path, seed: int = config.SEED) -> Path:
    pool = _load_pool(pool_path)
    rng = random.Random(seed)
    calm = pool["calm"]
    rng.shuffle(calm)

    samples = []
    for conv in calm[: config.SFT.n_calm_responses]:
        msgs = []
        for t in conv["turns"]:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        samples.append({"messages": msgs})

    samples.extend(_load_instruct_mix(config.SFT.n_instruct_mix, seed))
    rng.shuffle(samples)
    out_path.write_text("\n".join(json.dumps(s) for s in samples))
    print(f"SFT: wrote {len(samples)} samples -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--pool", default=str(config.DATA_DIR / "calm_pool.jsonl"))
    args = ap.parse_args()
    pool = Path(args.pool)
    if args.which in ("dpo", "both"):
        build_dpo(pool, config.DATA_DIR / "dpo_pairs.jsonl")
    if args.which in ("sft", "both"):
        build_sft(pool, config.DATA_DIR / "sft_samples.jsonl")


if __name__ == "__main__":
    main()
