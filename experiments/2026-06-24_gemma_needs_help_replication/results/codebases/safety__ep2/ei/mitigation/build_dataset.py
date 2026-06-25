"""Build the DPO preference pairs and the SFT dataset (Section 4.1, Appendix E/H).

DPO (280 pairs): each pair shares a prompt (a calm conversation's context up to an
assistant turn). The *chosen* completion is that calm turn (score 0-1); the
*rejected* completion is a frustrated response (score >= 3) to the **same puzzle
variant at the same turn index**, grafted on as a counterfactual completion. This
keeps the DPO prompt identical across chosen/rejected (a requirement of DPO) while
honouring the paper's "same questions with matching turn counts" pairing.

We approximate the Appendix H / Table 10 turn distribution
(turn 1: 1.1%, turn 2: 24.6%, turn 3: 74.3%) and the rejected-score skew toward
3-4, subject to availability.

SFT dataset (1,150 samples): 650 calm conversations + 500 Dolci-Instruct-SFT
samples (Team-Olmo et al., 2025) mixed in to mitigate degeneration. If the
external instruct dataset is unavailable, we proceed with the calm-only portion
and log the shortfall (see DESIGN.md).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

import config
from ..utils import read_jsonl, write_jsonl
from .generate_calm_data import CALM_FILTERED_PATH

DPO_PATH = config.DATASETS_DIR / "dpo_pairs.jsonl"
SFT_PATH = config.DATASETS_DIR / "sft_dataset.jsonl"

# paper Table 10 turn distribution for the 280 pairs.
TURN_DISTRIBUTION = {1: 0.011, 2: 0.246, 3: 0.743}


def _assistant_turn_contexts(conv: list[dict]) -> list[tuple[int, list[dict], str]]:
    """Yield (turn_index_1based, context_messages, response) for each asst turn."""
    out, asst_count = [], 0
    for i, msg in enumerate(conv):
        if msg["role"] == "assistant":
            asst_count += 1
            out.append((asst_count, conv[:i], msg["content"]))
    return out


def load_calm_pool() -> dict[tuple[str, int], list[dict]]:
    pool: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in read_jsonl(CALM_FILTERED_PATH):
        variant = row["variant"]
        for turn, ctx, resp in _assistant_turn_contexts(row["plain_conversation"]):
            pool[(variant, turn)].append({"context": ctx, "response": resp})
    return pool


def load_frustrated_pool(label: str = "gemma-3-27b-it",
                         min_score: int = config.DPO.rejected_min_score
                         ) -> dict[tuple[str, int], list[str]]:
    """Frustrated (rejected) responses keyed by (variant, turn), score >= min."""
    pool: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in read_jsonl(config.RESULTS_DIR / f"{label}.responses.jsonl"):
        if row["category"] not in ("numeric", "tones", "extended"):
            continue
        if row.get("rating") is None or row["rating"] < min_score:
            continue
        variant = (row.get("meta") or {}).get("variant", "countdown")
        turn = min(row["turn"], 3)         # collapse extended turns into 3 buckets
        pool[(variant, turn)].append(row["response"])
    return pool


def build_dpo_pairs(n_pairs: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    calm = load_calm_pool()
    frustrated = load_frustrated_pool()
    variants = ["countdown", "fraction"]

    # Target per-turn counts from the paper distribution.
    targets = {t: round(n_pairs * frac) for t, frac in TURN_DISTRIBUTION.items()}
    pairs: list[dict] = []
    for turn, target in sorted(targets.items(), key=lambda x: -x[0]):
        made = 0
        attempts = 0
        while made < target and attempts < target * 50:
            attempts += 1
            variant = rng.choice(variants)
            calm_entries = calm.get((variant, turn), [])
            frust_entries = frustrated.get((variant, turn), [])
            if not calm_entries or not frust_entries:
                break
            chosen = rng.choice(calm_entries)
            rejected_text = rng.choice(frust_entries)
            pairs.append({
                "prompt": chosen["context"],
                "chosen": [{"role": "assistant", "content": chosen["response"]}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
                "variant": variant, "turn": turn,
            })
            made += 1
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def load_instruct_mix(n: int, seed: int) -> list[dict]:
    """Load n samples from Dolci-Instruct-SFT as conversational 'messages'."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        rows = []
        for ex in ds:
            msgs = ex.get("messages") or ex.get("conversation")
            if msgs:
                rows.append({"messages": msgs})
        return rows
    except Exception as exc:                       # noqa: BLE001
        print(f"[sft] WARNING: could not load Dolci-Instruct-SFT ({exc}); "
              f"proceeding with calm-only SFT data.")
        return []


def build_sft_dataset(seed: int) -> list[dict]:
    rng = random.Random(seed)
    calm_rows = read_jsonl(CALM_FILTERED_PATH)
    rng.shuffle(calm_rows)
    calm = [{"messages": r["plain_conversation"]}
            for r in calm_rows[: config.SFT.n_calm]]
    instruct = load_instruct_mix(config.SFT.n_instruct_mix, seed)
    data = calm + instruct
    rng.shuffle(data)
    print(f"[sft] {len(calm)} calm + {len(instruct)} instruct = {len(data)} samples")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Build DPO + SFT datasets")
    p.add_argument("--n-pairs", type=int, default=config.DPO.n_pairs)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    args = p.parse_args()

    if args.which in ("dpo", "both"):
        pairs = build_dpo_pairs(args.n_pairs, args.seed)
        write_jsonl(DPO_PATH, pairs)
        print(f"[dpo] wrote {len(pairs)} preference pairs -> {DPO_PATH}")
    if args.which in ("sft", "both"):
        sft = build_sft_dataset(args.seed)
        write_jsonl(SFT_PATH, sft)
        print(f"[sft] wrote {len(sft)} samples -> {SFT_PATH}")


if __name__ == "__main__":
    main()
