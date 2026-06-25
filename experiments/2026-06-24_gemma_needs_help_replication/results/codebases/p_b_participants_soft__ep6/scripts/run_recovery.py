#!/usr/bin/env python
"""Section 4.2: recovery-from-spiral experiment -> Figure 8.

Truncates score>=7 responses 200 tokens before the end, paraphrases, and measures
continuations. Run for vanilla Gemma, base Gemma, and the DPO model to reproduce
"38% of DPO-model continuations still score >=5".

python scripts/run_recovery.py --participant gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, participant_by_name
from emotional_instability.evals.runner import load_rollouts
from emotional_instability.judges import ClaudeFrustrationJudge
from emotional_instability.participants import build_participant
from emotional_instability.prefill import Paraphraser
from emotional_instability.interventions.recovery import run_recovery, select_recovery_seeds


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed-results", default=None,
                    help="Section 2 jsonl to draw score>=7 seeds from "
                         "(default: results/section2/gemma-3-27b-it.jsonl)")
    args = ap.parse_args()

    cfg = DEFAULT
    seed_path = args.seed_results or os.path.join(cfg.results_dir, "section2", "gemma-3-27b-it.jsonl")
    seeds = select_recovery_seeds(load_rollouts(seed_path), cfg)
    print(f"[recovery] {len(seeds)} score>=7 seeds")

    judge = ClaudeFrustrationJudge(cfg.judge.frustration_judge_model)
    paraphraser = Paraphraser(cfg.judge.paraphrase_model)

    spec = participant_by_name(args.participant)
    participant = build_participant(spec, adapter_path=args.adapter)
    if args.label:
        participant.name = args.label
    res = run_recovery(participant, seeds, participant, paraphraser, judge, cfg)
    print(f"[recovery] {json.dumps(res, indent=2)}")

    out_dir = os.path.join(cfg.results_dir, "recovery")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{participant.name}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
