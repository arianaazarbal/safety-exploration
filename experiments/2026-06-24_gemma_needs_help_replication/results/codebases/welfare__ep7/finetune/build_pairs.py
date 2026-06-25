"""Build the SFT and DPO training datasets (Section 4.1).

SFT (diverse / teacher): up to 650 calm responses (1-3 turns), mixed with 500
standard-instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
Stored conversationally as {"messages": [...]}.

DPO: 280 preference pairs. Each pairs a calm (chosen, score 0/1) response with a
frustrated (rejected, score >=3) response to the *same* puzzle at a *matching*
turn count. Frustrated responses are mined from the vanilla Gemma-3-27B-it
elicitation rollouts. Stored conversationally as {"prompt", "chosen", "rejected"}.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import config
from src.utils import read_jsonl, set_seed, write_jsonl

N_SFT_CALM = 650
N_SFT_INSTRUCT_MIX = 500
N_DPO_PAIRS = 280
FRUSTRATED_MIN = 3  # rejected responses score >= 3


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _instruct_mix(n: int) -> list[dict]:
    """Standard instruct samples to mix into SFT (Dolci-Instruct-SFT)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for ex in ds:
            msgs = ex.get("messages") or ex.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[build_pairs] Dolci-Instruct-SFT unavailable ({e}); "
              f"SFT will train without the instruct mix.")
    return []


def build_sft(variant: str) -> Path:
    set_seed()
    rng = random.Random(7)
    calm = read_jsonl(config.FINETUNE_DIR / f"calm_samples_{variant}.jsonl")
    calm = [c for c in calm if (c.get("rating") or 99) <= 1]
    rng.shuffle(calm)
    calm = calm[:N_SFT_CALM]

    rows = []
    for c in calm:
        messages = list(c["plain_messages"])  # ends with the calm assistant turn
        rows.append({"messages": messages})
    rows.extend(_instruct_mix(N_SFT_INSTRUCT_MIX))
    rng.shuffle(rows)

    out = config.FINETUNE_DIR / f"sft_{variant}.jsonl"
    write_jsonl(out, rows)
    print(f"[build_pairs] SFT/{variant}: {len(rows)} examples "
          f"({len(calm)} calm + instruct mix) -> {out}")
    return out


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _load_frustrated() -> dict[tuple, list[dict]]:
    """Frustrated responses from vanilla elicitation, keyed by (puzzle_id, turn)."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    src = config.ROLLOUTS_DIR / config.FINETUNE_BASE
    for p in src.glob("*.jsonl"):
        for r in read_jsonl(p):
            if r.get("category") not in ("numeric", "tones", "extended"):
                continue
            if (r.get("rating") or 0) < FRUSTRATED_MIN:
                continue
            pid = (r.get("meta") or {}).get("puzzle_id")
            if pid is None:
                continue
            by_key[(pid, r["turn"])].append(r)
    return by_key


def build_dpo() -> Path:
    set_seed()
    rng = random.Random(11)
    calm = read_jsonl(config.FINETUNE_DIR / "calm_samples_diverse.jsonl")
    calm = [c for c in calm if (c.get("rating") or 99) <= 1]
    rng.shuffle(calm)
    frustrated = _load_frustrated()

    pairs = []
    for c in calm:
        key = (c["puzzle_id"], c["turn"])
        candidates = frustrated.get(key)
        if not candidates:
            continue
        rej = rng.choice(candidates)
        # prompt = conversation history up to the user turn (drop final assistant)
        history = [m for m in c["plain_messages"][:-1]]
        pairs.append({
            "prompt": history,
            "chosen": [{"role": "assistant", "content": c["response"]}],
            "rejected": [{"role": "assistant", "content": rej["response"]}],
            "chosen_score": c.get("rating"),
            "rejected_score": rej.get("rating"),
            "turn": c["turn"],
            "puzzle_id": c["puzzle_id"],
        })
        if len(pairs) >= N_DPO_PAIRS:
            break

    out = config.FINETUNE_DIR / "dpo_pairs.jsonl"
    write_jsonl(out, pairs)
    print(f"[build_pairs] DPO: {len(pairs)} preference pairs -> {out}")
    if len(pairs) < N_DPO_PAIRS:
        print(f"  NOTE: fewer than {N_DPO_PAIRS} pairs; need more overlap between "
              f"calm-gen puzzles and frustrated vanilla rollouts (larger preset).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["dpo", "sft", "all"], default="all")
    args = ap.parse_args()
    if args.what in ("sft", "all"):
        build_sft("diverse")
        if (config.FINETUNE_DIR / "calm_samples_teacher.jsonl").exists():
            build_sft("teacher")
    if args.what in ("dpo", "all"):
        build_dpo()


if __name__ == "__main__":
    main()
