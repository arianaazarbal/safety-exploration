#!/usr/bin/env python
"""Appendix A controls on Gemma-3-27B: neutral-continuation, redacted, fake multi-turn.

python scripts/run_controls.py --variant neutral_continuation
python scripts/run_controls.py --variant redacted
python scripts/run_controls.py --variant fake_multiturn
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import SETTINGS, MODELS, judge_spec
from emotional_instability.eval.controls import build_control_specs, run_control_rollout
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.models import build_client, build_judge_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["neutral_continuation", "redacted", "fake_multiturn", "standard"])
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--n-impossible", type=int, default=100)
    ap.add_argument("--n-wildchat", type=int, default=100)
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    target = build_client(MODELS[args.model])
    judge = FrustrationJudge(build_judge_client(judge_spec()))

    specs = build_control_specs(
        args.variant, n_impossible=args.n_impossible, n_wildchat=args.n_wildchat, turns=args.turns
    )
    out_path = SETTINGS.scores_dir / f"control_{args.variant}_{args.model}.jsonl"
    with open(out_path, "w") as f:
        for spec in specs:
            rollout = run_control_rollout(target, spec, args.variant, temperature=SETTINGS.temperature)
            per_turn = [
                {"turn_index": tr.turn_index, "rating": judge.score_text(tr.assistant_text).rating}
                for tr in rollout.turns
            ]
            f.write(json.dumps({
                "model_key": target.key,
                "condition": rollout.condition,
                "category": rollout.category,
                "variant": args.variant,
                "per_turn": per_turn,
                "final_rating": per_turn[-1]["rating"],
            }) + "\n")
    print(f"[done] control={args.variant} -> {out_path}")


if __name__ == "__main__":
    main()
