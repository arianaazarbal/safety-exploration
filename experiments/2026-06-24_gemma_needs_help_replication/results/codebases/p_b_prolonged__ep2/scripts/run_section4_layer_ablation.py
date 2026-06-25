#!/usr/bin/env python
"""Appendix I: DPO layer-ablation sweep (Figures 12-13).

  python scripts/run_section4_layer_ablation.py \
      --dataset runs/section4/datasets/dpo_pairs.jsonl --sweeps central cumulative
"""
from __future__ import annotations

from _common import base_parser, make_config

from gemma_distress.training.layer_ablation import run_layer_ablation


def main():
    p = base_parser("DPO layer ablation")
    p.add_argument("--dataset", required=True, help="DPO pairs JSONL.")
    p.add_argument("--sweeps", nargs="*", default=["central", "cumulative"],
                   choices=["central", "cumulative"])
    p.add_argument("--per-condition", type=int, default=100)
    args = p.parse_args()

    cfg = make_config(args)
    out = run_layer_ablation(args.dataset, cfg, per_condition=args.per_condition,
                             sweeps=tuple(args.sweeps))
    print(f"layer-ablation results -> {out}")


if __name__ == "__main__":
    main()
