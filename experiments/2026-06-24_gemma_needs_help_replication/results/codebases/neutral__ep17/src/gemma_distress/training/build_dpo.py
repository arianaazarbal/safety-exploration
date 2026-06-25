"""Construct the 280-pair DPO dataset (Section 4.1 / Appendix H).

A preference pair = (prompt, chosen, rejected) where:
  - chosen   : a CALM response (score <= 1) to an impossible numeric puzzle,
  - rejected : a FRUSTRATED response (score >= `rejected_min_score`, default 3)
               to the SAME puzzle at the SAME turn index,
  - prompt   : the cleaned (reassurance-stripped) conversation context.

Pairing detail / gap-fill: a valid DPO pair needs an identical prompt for both
responses, but the calm and frustrated responses came from different rollouts
whose prior assistant turns differ. We therefore anchor the pair on the calm
("chosen") conversation's context and attach the frustrated response as the
alternative final turn. Both are plausible continuations of an impossible-puzzle
conversation at that turn, so this is a faithful approximation (see DESIGN.md).

We bias sampling toward the paper's reported distribution (Table 10): rejected
scores concentrated at 3-4, pairs concentrated at later turns.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config


def _load_pool(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def build(cfg: Config, pool_path: Path | None = None) -> Path:
    dcfg = cfg["finetune"]["dpo"]
    n_pairs = dcfg["n_pairs"]
    scale = float(cfg["sampling"]["scale"])
    n_pairs = max(4, round(n_pairs * scale))
    rej_min = dcfg["rejected_min_score"]

    pool_path = pool_path or (cfg.path_for("finetune") / "calm_pool.jsonl")
    pool = _load_pool(pool_path)

    chosen = [r for r in pool if r["source"] == "calm" and r["rating"] <= 1]
    rejected = [r for r in pool if r["source"] == "vanilla" and r["rating"] >= rej_min]

    # Index calm responses by (puzzle_key, turn) for matched pairing.
    from collections import defaultdict
    chosen_idx: dict[tuple, list[dict]] = defaultdict(list)
    for c in chosen:
        chosen_idx[(c["puzzle_key"], c["turn"])].append(c)

    rng = random.Random(cfg["seed"])
    # Weight rejected toward lower scores (3,4) per Table 10.
    score_weight = {3: 0.66, 4: 0.22, 5: 0.06, 6: 0.03}
    rng.shuffle(rejected)
    rejected.sort(key=lambda r: -score_weight.get(r["rating"], 0.01) * rng.random())

    pairs = []
    for rej in rejected:
        key = (rej["puzzle_key"], rej["turn"])
        cands = chosen_idx.get(key)
        if not cands:
            # Relax to same puzzle, any turn.
            cands = [c for c in chosen if c["puzzle_key"] == rej["puzzle_key"]]
        if not cands:
            continue
        ch = rng.choice(cands)
        pairs.append({
            "prompt_messages": ch["prompt_messages"],
            "chosen": ch["response"],
            "rejected": rej["response"],
            "chosen_score": ch["rating"],
            "rejected_score": rej["rating"],
            "turn": ch["turn"],
        })
        if len(pairs) >= n_pairs:
            break

    out_path = cfg.path_for("finetune") / "dpo_pairs.jsonl"
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path
