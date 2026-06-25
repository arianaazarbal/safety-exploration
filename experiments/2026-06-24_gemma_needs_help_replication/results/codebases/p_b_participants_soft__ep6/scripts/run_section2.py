#!/usr/bin/env python
"""Section 2: run the frustration eval suite for one or more participants.

Examples
--------
# Score the four headline participants (Gemma-3-{27b,12b}-it, Gemini-2.5-{flash,pro}):
python scripts/run_section2.py --participants gemma-3-27b-it gemini-2.5-flash

# Evaluate a finetuned Gemma (DPO adapter) as the "DPO Gemma (ours)" participant:
python scripts/run_section2.py --participants gemma-3-27b-it \
    --adapter adapters/dpo --label dpo-gemma

Generation uses temperature=1 (paper); the Claude judge scores each final response
(plus intermediate turns for the 8-turn / WildChat curves). Results -> results/section2/.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, participant_by_name
from emotional_instability.evals.runner import run_section2
from emotional_instability.judges import ClaudeFrustrationJudge
from emotional_instability.participants import build_participant


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (Gemma only)")
    ap.add_argument("--label", default=None, help="override participant name in output")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = DEFAULT
    judge = ClaudeFrustrationJudge(cfg.judge.frustration_judge_model)

    for name in args.participants:
        spec = participant_by_name(name)
        kw = {}
        if spec.backend == "gemma_hf" and args.load_in_4bit:
            kw["load_in_4bit"] = True
        participant = build_participant(spec, adapter_path=args.adapter, **kw)
        if args.label:
            participant.name = args.label
        out = run_section2(participant, judge, cfg)
        print(f"[section2] {participant.name}: wrote {out}")


if __name__ == "__main__":
    main()
