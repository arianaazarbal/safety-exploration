#!/usr/bin/env python
"""Section 4.1: generate calm/vanilla paired data and build SFT + DPO datasets.

python scripts/generate_calm_data.py --n-puzzles 600

Writes:
  data/calm/paired.jsonl     -- raw calm+vanilla conversations with per-turn scores
  data/calm/sft.jsonl        -- 650 calm examples + 500 Dolci mix (chat format)
  data/calm/dpo.jsonl        -- 280 (prompt, chosen, rejected) preference pairs
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, INTERVENTION_BASE_MODEL
from emotional_instability.interventions.calm_data import generate_paired_data, save_paired_data
from emotional_instability.interventions.dataset import (
    build_dpo_pairs,
    build_sft_examples,
    load_dolci_mix,
)
from emotional_instability.judges import ClaudeFrustrationJudge
from emotional_instability.participants import build_participant


def _write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-puzzles", type=int, default=800,
                    help="paired conversations to generate (need >=650 calm + 280 frustrated)")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = DEFAULT
    participant = build_participant(INTERVENTION_BASE_MODEL, load_in_4bit=args.load_in_4bit)
    judge = ClaudeFrustrationJudge(cfg.judge.frustration_judge_model)

    records = generate_paired_data(participant, judge, cfg, args.n_puzzles)
    paired_path = os.path.join(cfg.data_dir, "calm", "paired.jsonl")
    save_paired_data(records, paired_path)
    print(f"[calm] wrote {len(records)} paired records -> {paired_path}")

    sft = build_sft_examples(records, cfg) + load_dolci_mix(cfg)
    _write(sft, os.path.join(cfg.data_dir, "calm", "sft.jsonl"))
    dpo = build_dpo_pairs(records, cfg)
    _write(dpo, os.path.join(cfg.data_dir, "calm", "dpo.jsonl"))
    print(f"[calm] SFT examples: {len(sft)}  DPO pairs: {len(dpo)}")


if __name__ == "__main__":
    main()
