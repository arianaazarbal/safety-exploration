"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

Pairs a frustrated (rejected, score >=3) response with a calm (chosen, score
0/1) response to the same puzzle at the same turn count. Target size 280 with a
turn/score distribution close to Table 10 (biased to middle scores and later
turns, which is what naturally arises).

Each output record is {prompt: messages, chosen: str, rejected: str}, where
`prompt` is the (plain, additions-stripped) conversation context. Both members
of a pair share this prompt.

GAP-FILLING CHOICE: chosen and rejected come from different rollouts, so they do
not share an identical earlier-assistant history. We use the rejected sample's
plain context as the shared prompt and graft the calm final turn from a matching
calm rollout. This is the standard pragmatic construction when paired
same-context generations are not available; see DESIGN.md.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

# Approximate target distributions from Table 10.
TARGET_SCORE_DIST = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}
TARGET_TURN_DIST = {1: 0.011, 2: 0.246, 3: 0.743}
N_PAIRS = 280


@dataclass
class CalmRejectedPools:
    # key (puzzle_id, turn_index) -> list of records
    calm: dict
    rejected: dict


def _load_pools(calm_raw_path: str) -> CalmRejectedPools:
    calm = defaultdict(list)
    rejected = defaultdict(list)
    with open(calm_raw_path, encoding="utf-8") as fh:
        for line in fh:
            conv = json.loads(line)
            puzzle = conv["puzzle_id"]
            for t in conv["turns"]:
                key = (puzzle, t["turn_index"])
                rec = {
                    "response": t["response"],
                    "score": t["score"],
                    "context": t["plain_context"],
                    "turn": t["turn_index"] + 1,  # 1-based to match Table 10
                }
                if conv["all_calm"] and t["score"] <= 1:
                    calm[key].append(rec)
                if t["score"] >= 3:
                    rejected[key].append(rec)
    return CalmRejectedPools(calm=calm, rejected=rejected)


def build(calm_raw_path: str, out_path: str, n_pairs: int = N_PAIRS, seed: int = 0) -> str:
    rng = random.Random(seed)
    pools = _load_pools(calm_raw_path)

    # Collect all candidate rejected records that have a matching calm partner.
    candidates = []
    for key, rejs in pools.rejected.items():
        if key not in pools.calm or not pools.calm[key]:
            continue
        for r in rejs:
            candidates.append((key, r))

    # Weight selection toward the Table 10 score distribution where possible.
    def weight(rec):
        s = min(7, rec["score"])
        return TARGET_SCORE_DIST.get(s, 0.029) * TARGET_TURN_DIST.get(rec["turn"], 0.1)

    rng.shuffle(candidates)
    candidates.sort(key=lambda kr: -weight(kr[1]))

    pairs = []
    used_calm = defaultdict(set)
    for key, rej in candidates:
        if len(pairs) >= n_pairs:
            break
        calm_options = [
            c for i, c in enumerate(pools.calm[key]) if i not in used_calm[key]
        ]
        if not calm_options:
            calm_options = pools.calm[key]  # allow reuse if exhausted
        chosen = rng.choice(calm_options)
        pairs.append({
            "prompt": rej["context"],          # shared plain context (messages)
            "chosen": chosen["response"],
            "rejected": rej["response"],
            "rejected_score": rej["score"],
            "turn": rej["turn"],
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path
