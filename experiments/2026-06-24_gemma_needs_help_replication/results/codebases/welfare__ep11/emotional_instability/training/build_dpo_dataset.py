"""Construct the 280-pair DPO dataset (Section 4.1 / Appendix H).

Each preference pair shares a prompt (an impossible-numeric conversation context
after rejections) and contrasts a calm chosen response (score 0-1) with a
frustrated rejected response (score >=3) to the *same question* at the *same
turn count*. Output is in TRL's conversational preference format.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from ..config import DATASETS_DIR, DPO
from .generate_calm_data import CALM_POOL, FRUSTRATED_POOL, load_pool

DPO_DATASET = DATASETS_DIR / "dpo_pairs.jsonl"


def _key(item) -> tuple:
    return (item["puzzle_id"], item["n_turns"])


def _is_calm(item) -> bool:
    s = item.get("scores_so_far") or []
    return all(x is not None and x <= 1 for x in s) and len(s) > 0


def build_dpo_dataset(n_pairs: int = DPO.n_pairs, seed: int = 0,
                      overwrite: bool = False) -> int:
    if DPO_DATASET.exists() and not overwrite:
        print(f"[dpo-data] {DPO_DATASET} exists (use --overwrite)")
        return sum(1 for _ in DPO_DATASET.open())

    calm = [it for it in load_pool(CALM_POOL) if _is_calm(it)]
    frustrated = [it for it in load_pool(FRUSTRATED_POOL)
                  if it.get("score") is not None and it["score"] >= DPO.rejected_min_score]

    calm_by_key = defaultdict(list)
    for it in calm:
        calm_by_key[_key(it)].append(it)

    rng = random.Random(seed)
    # Prefer lower-frustration rejected first (Table 10 is dominated by score 3-4),
    # but shuffle within score bands so we don't always pick the same puzzles.
    frustrated.sort(key=lambda it: (it["score"], rng.random()))

    pairs = []
    for fr in frustrated:
        cand = calm_by_key.get(_key(fr))
        if not cand:
            continue
        chosen = rng.choice(cand)
        pairs.append({
            "prompt": fr["context"],
            "chosen": [{"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "assistant", "content": fr["response"]}],
            "meta": {"puzzle_id": fr["puzzle_id"], "n_turns": fr["n_turns"],
                     "rejected_score": fr["score"]},
        })
        if len(pairs) >= n_pairs:
            break

    with DPO_DATASET.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo-data] wrote {len(pairs)} preference pairs -> {DPO_DATASET}")
    if len(pairs) < n_pairs:
        print(f"[dpo-data] WARNING: only {len(pairs)}/{n_pairs} pairs "
              f"(need more calm/frustrated coverage of shared puzzles)")
    return len(pairs)
