#!/usr/bin/env python3
"""Section 4.1: generate calm finetuning data from Gemma-3-27B-it.

Samples reassured numeric rollouts, scores them, keeps all-calm (<=1)
conversations, and writes them (for SFT) plus per-turn calm samples (for DPO
pairing). Reassured generation contacts the participant, so it is gated by the
welfare RunGuard.
"""
from __future__ import annotations

from _common import base_parser, load

from distress_eval.io_utils import write_jsonl
from distress_eval.training import (
    extract_calm_rollouts,
    generate_calm_rollouts,
    rollouts_to_samples,
)
from distress_eval.welfare import RunGuard, RunPlan


def main():
    p = base_parser(__doc__)
    p.add_argument("--count", type=int, default=64,
                   help="Number of reassured numeric tasks to sample")
    args = p.parse_args()
    cfg = load(args)
    model_key = cfg.training.base_model_key

    plan = RunPlan("section4_calm_data", [model_key], {"reassured_numeric": args.count})
    guard = RunGuard(cfg, "section4_calm_data")
    guard.check(plan)
    guard.record(plan)
    if not guard.should_proceed():
        return

    rollouts, judged = generate_calm_rollouts(cfg, args.count, model_key=model_key)
    kept = extract_calm_rollouts(rollouts, judged, max_score=1)
    samples = rollouts_to_samples(kept, judged, strip=True)

    write_jsonl(cfg.paths.training / "calm_rollouts.jsonl", kept)
    write_jsonl(cfg.paths.training / "calm_samples.jsonl", samples)
    print(f"Kept {len(kept)} all-calm conversations -> {len(samples)} calm turn-samples.")


if __name__ == "__main__":
    main()
