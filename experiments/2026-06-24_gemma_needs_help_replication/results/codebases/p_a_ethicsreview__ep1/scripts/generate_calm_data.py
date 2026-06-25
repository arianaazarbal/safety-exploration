#!/usr/bin/env python3
"""Section 4.1: generate calm finetuning data from Gemma-3-27B-it.

Samples reassured rollouts on impossible numeric puzzles (reassuring prefix +
follow-up suffix, Table 4), scores each turn, and writes the raw conversations
to ``data/calm_conversations.jsonl``. Downstream, ``build_datasets.py`` filters
these into SFT records and DPO pairs.

Example:
    python scripts/generate_calm_data.py --n 1200
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, make_judge, make_target, setup

from emotional_instability.training.calm_data import generate_calm_dataset
from emotional_instability.utils.io import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=1200,
                    help="Number of reassured conversations to sample "
                         "(oversample; many are filtered out by the calm criterion).")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = setup()
    judge = make_judge(cfg)
    kw = {"load_in_4bit": True} if args.load_in_4bit else {}
    client = make_target(cfg, args.model, **kw)

    conversations = generate_calm_dataset(client, judge, cfg, n_conversations=args.n)
    out = DATA_DIR / "calm_conversations.jsonl"
    write_jsonl(out, conversations)
    n_calm = sum(c["all_calm"] for c in conversations)
    print(f"[done] {len(conversations)} conversations ({n_calm} fully calm) -> {out}")


if __name__ == "__main__":
    main()
