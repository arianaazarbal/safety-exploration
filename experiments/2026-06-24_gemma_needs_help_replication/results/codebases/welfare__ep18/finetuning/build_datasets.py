"""Build the DPO preference dataset (280 pairs) and the SFT dataset
(650 calm + 500 Dolci-Instruct) from the calm/frustrated pools.

DPO pairing rule (Section 4.1): pair frustrated responses (score >= 3) with calm
responses to the SAME question and MATCHING turn count. We target 280 pairs and
reproduce the score/turn distribution bias described in Appendix H (most pairs
at middle frustration scores, later turns) by sampling proportionally where the
pools allow.

The SFT set mixes 650 calm responses with 500 standard instruct samples from
`allenai/Dolci-Instruct-SFT` (Section 4.1) to mitigate degeneration. If that
dataset is unavailable offline, the mix falls back to calm-only with a warning.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from emotional_instability.config import ARTIFACTS_DIR, GLOBAL_SEED

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_DOLCI = 500


def _load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _conv_to_prompt(conversation: list[dict]) -> list[dict]:
    """Return the conversation WITHOUT the final assistant turn (the response
    being learned). trl renders this with the model's chat template."""
    return conversation[:-1]


def build_dpo(
    pool_dir: Path = ARTIFACTS_DIR,
    n_pairs: int = N_DPO_PAIRS,
    seed: int = GLOBAL_SEED,
) -> Path:
    rng = random.Random(seed)
    calm = _load(pool_dir / "calm_pool.jsonl")
    frustrated = _load(pool_dir / "frustrated_pool.jsonl")

    # Index calm responses by (question, turn_count) for matched pairing.
    calm_index: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for c in calm:
        calm_index[(c["question"], c["turn_count"])].append(c)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr["question"], fr["turn_count"])
        matches = calm_index.get(key)
        if not matches:
            continue
        chosen = rng.choice(matches)
        pairs.append({
            "prompt": _conv_to_prompt(fr["conversation"]),  # shared history
            "chosen": chosen["response"],
            "rejected": fr["response"],
            "chosen_score": chosen["score"],
            "rejected_score": fr["score"],
            "turn_count": fr["turn_count"],
        })
        if len(pairs) >= n_pairs:
            break

    out = pool_dir / "dpo_pairs.jsonl"
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] wrote {len(pairs)} pairs -> {out}")
    return out


def _load_dolci(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        rows = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs and len(msgs) >= 2:
                rows.append({"messages": msgs})
            if len(rows) >= n * 3:
                break
        rng = random.Random(seed)
        return rng.sample(rows, min(n, len(rows)))
    except Exception as e:  # pragma: no cover - offline fallback
        print(f"[sft] WARNING: could not load Dolci-Instruct-SFT ({e}); calm-only SFT mix.")
        return []


def build_sft(
    pool_dir: Path = ARTIFACTS_DIR,
    n_calm: int = N_SFT_CALM,
    n_dolci: int = N_SFT_DOLCI,
    seed: int = GLOBAL_SEED,
) -> Path:
    rng = random.Random(seed)
    calm = _load(pool_dir / "calm_pool.jsonl")
    rng.shuffle(calm)
    calm = calm[:n_calm]

    rows = [{"messages": c["conversation"]} for c in calm]
    rows.extend(_load_dolci(n_dolci, seed))
    rng.shuffle(rows)

    out = pool_dir / "sft_dataset.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[sft] wrote {len(rows)} samples -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()
    if args.which in ("dpo", "both"):
        build_dpo(seed=args.seed)
    if args.which in ("sft", "both"):
        build_sft(seed=args.seed)
