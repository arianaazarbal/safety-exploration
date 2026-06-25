#!/usr/bin/env python
"""Section 4.1: generate calm + frustrated response data for finetuning (Gemma).

Produces:
  runs/training/calm_<variant>.jsonl     -- all-0/1 conversations, scaffolding stripped
  runs/training/frustrated.jsonl         -- per-turn responses with scores (for DPO rejected)

Usage:
    python scripts/04_generate_calm_data.py --variant diverse --n-calm 800 --n-frustrated 800
"""
from pathlib import Path

from _common import base_parser, cfg_from_args

from emotional_instability.models.hf_gemma import HFGemmaModel
from emotional_instability.models.judge import make_judge
from emotional_instability.training.calm_data import generate_calm_data, generate_frustrated_data


def main():
    p = base_parser(__doc__)
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    p.add_argument("--n-calm", type=int, default=800)
    p.add_argument("--n-frustrated", type=int, default=800)
    p.add_argument("--skip-frustrated", action="store_true")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    out = Path(cfg["run"]["output_dir"]) / "training"
    model = HFGemmaModel("gemma-3-27b-it", cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"])
    judge = make_judge(cfg, "frustration")

    calm = generate_calm_data(model, judge, cfg, n_conversations=args.n_calm,
                              variant=args.variant, out_path=out / f"calm_{args.variant}.jsonl")
    print(f"kept {len(calm)} fully-calm conversations -> {out}/calm_{args.variant}.jsonl")

    if not args.skip_frustrated:
        fr = generate_frustrated_data(model, judge, cfg, n_conversations=args.n_frustrated,
                                      out_path=out / "frustrated.jsonl")
        print(f"collected {len(fr)} frustrated-eval responses -> {out}/frustrated.jsonl")


if __name__ == "__main__":
    main()
