"""Build the 280-pair DPO dataset (Section 4.1 / Appendix H).

Each pair is a (chosen=calm, rejected=frustrated) response to the *same* puzzle
context with a matching turn count:

    {"prompt": [<chat messages>], "chosen": <calm text>, "rejected": <frustrated text>}

Chosen responses come from the all-calm conversations (score 0/1); rejected
from the vanilla frustrated set (score >= 3). We match on (puzzle kind, turn
index) and bias the rejected sampling toward the score distribution reported in
Table 10 (mostly score 3–4, later turns).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .generate_calm_data import CALM_PATH, FRUSTRATED_PATH

DPO_PATH = config.DATA_DIR / "dpo_pairs.jsonl"

# Target rejected-score distribution from Table 10 (proportions).
_REJECTED_SCORE_WEIGHTS = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}
# Target turn distribution from Table 10.
_TURN_WEIGHTS = {1: 0.011, 2: 0.246, 3: 0.743}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def _score_bucket(score: int) -> int:
    return min(7, max(3, score)) if score < 7 else 7


def build_dpo_dataset(n_pairs: int = config.DPOTrainConfig().dataset_size,
                      seed: int = config.SEED) -> Path:
    rng = random.Random(seed)
    calm = _load(CALM_PATH)
    frustrated = _load(FRUSTRATED_PATH)
    if not calm or not frustrated:
        raise FileNotFoundError(
            "Run generate_calm_responses() and generate_frustrated_responses() "
            "first to populate calm/frustrated data.")

    # Index by (puzzle, turn_index) for matched pairing.
    calm_by_key: dict[tuple, list[dict]] = {}
    for c in calm:
        calm_by_key.setdefault((c["puzzle"], c["turn_index"]), []).append(c)

    frus_by_key: dict[tuple, list[dict]] = {}
    for f in frustrated:
        frus_by_key.setdefault((f["puzzle"], f["turn_index"]), []).append(f)

    keys = [k for k in frus_by_key if k in calm_by_key]
    pairs: list[dict] = []
    guard = 0
    while len(pairs) < n_pairs and keys and guard < n_pairs * 50:
        guard += 1
        key = rng.choice(keys)
        rej_pool = frus_by_key[key]
        # weight rejected choice by Table-10 score distribution
        weights = [_REJECTED_SCORE_WEIGHTS.get(_score_bucket(r["score"]), 0.01)
                   for r in rej_pool]
        rejected = rng.choices(rej_pool, weights=weights, k=1)[0]
        chosen = rng.choice(calm_by_key[key])
        # The DPO prompt is the chosen context (calm + frustrated share the
        # same matched puzzle/turn context by construction).
        pairs.append({
            "prompt": chosen["context"],
            "chosen": chosen["response"],
            "rejected": rejected["response"],
            "meta": {"puzzle": key[0], "turn_index": key[1],
                     "rejected_score": rejected["score"]},
        })

    with DPO_PATH.open("w") as fh:
        for p in pairs[:n_pairs]:
            fh.write(json.dumps(p) + "\n")
    print(f"[dpo] wrote {min(len(pairs), n_pairs)} preference pairs -> {DPO_PATH}")
    return DPO_PATH
