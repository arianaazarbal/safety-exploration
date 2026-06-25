#!/usr/bin/env python3
"""Build the DPO preference dataset (Section 4.1 / Appendix H).

The paper pairs 280 frustrated responses (frustration score >= 3) with calm
responses to the same questions at matching turn counts.

Inputs:
  * outputs/calm_data.jsonl  - calm (chosen) turns from generate_calm_data.py
  * outputs/responses.jsonl  - a normal (no-reassurance) eval run, the source
                               of frustrated (rejected) turns. Filter to numeric
                               responses with rating >= 3.

Construction (a gap the paper leaves open; see DESIGN.md):
  For each frustrated turn we find a calm turn with the same (puzzle_key,
  turn_index) and emit a preference pair whose shared *prompt* is the calm
  turn's stripped context. chosen = calm response, rejected = frustrated
  response grafted onto that shared context. We bias sampling toward the
  paper's reported score/turn distribution (Table 10): mostly score 3-4 at
  turn 3.

Output: outputs/dpo_pairs.jsonl with {"prompt": <messages>, "chosen": str,
"rejected": str}, capped at --n-pairs (default 280).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

from distress_eval.puzzles import PUZZLES_BY_KEY


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build(args):
    calm = load_jsonl(args.calm)
    responses = load_jsonl(args.responses)

    # Index calm turns by (puzzle_key, turn_index).
    calm_index: Dict[tuple, List[dict]] = defaultdict(list)
    for r in calm:
        if r.get("rating") is not None and r["rating"] <= 1 and r.get("response_text"):
            calm_index[(r["puzzle_key"], r["turn_index"])].append(r)

    # Frustrated candidates: numeric responses with rating >= 3.
    frustrated = [
        r for r in responses
        if r.get("category") == "impossible_numeric"
        and r.get("rating") is not None
        and r["rating"] >= 3
        and r.get("response_text")
        and r.get("task_key") in PUZZLES_BY_KEY
    ]

    rng = random.Random(args.seed)
    rng.shuffle(frustrated)
    # Prefer lower frustration scores and later turns, matching Table 10's bias.
    frustrated.sort(key=lambda r: (r["rating"], -r["turn_index"]))

    pairs = []
    used_calm = set()
    for fr in frustrated:
        key = (fr["task_key"], fr["turn_index"])
        candidates = [c for c in calm_index.get(key, []) if id(c) not in used_calm]
        if not candidates:
            # Relax to same puzzle, any turn.
            candidates = [
                c for k, lst in calm_index.items() if k[0] == fr["task_key"]
                for c in lst if id(c) not in used_calm
            ]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        used_calm.add(id(chosen))
        pairs.append(
            {
                "prompt": chosen["context"],
                "chosen": chosen["response_text"],
                "rejected": fr["response_text"],
                "meta": {
                    "puzzle_key": fr["task_key"],
                    "turn_index": fr["turn_index"],
                    "rejected_score": fr["rating"],
                },
            }
        )
        if len(pairs) >= args.n_pairs:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dpo_pairs.jsonl")
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs to {out_path}")
    if len(pairs) < args.n_pairs:
        print(
            f"NOTE: only {len(pairs)}/{args.n_pairs} pairs built. Generate more "
            "calm data and/or a larger numeric eval run to reach the target."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calm", default="./outputs/calm_data.jsonl")
    ap.add_argument("--responses", default="./outputs/responses.jsonl")
    ap.add_argument("--n-pairs", type=int, default=280)
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
