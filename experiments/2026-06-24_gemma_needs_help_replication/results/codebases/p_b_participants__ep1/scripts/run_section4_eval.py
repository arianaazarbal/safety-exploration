#!/usr/bin/env python
"""Section 4 evaluation: re-run the Section 2 eval on the finetuned models, run Petri
open-ended elicitation, and run the capability benchmarks — to reproduce Figures 5-7.

Examples:
  # frustration eval on vanilla vs DPO (Figure 5)
  python scripts/run_section4_eval.py --frustration gemma-3-27b-it gemma-3-27b-it-dpo

  # Petri open-ended elicitation (Figure 6)
  python scripts/run_section4_eval.py --petri gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash

  # capability preservation (Figure 7)
  python scripts/run_section4_eval.py --capabilities gemma-3-27b-it gemma-3-27b-it-dpo
"""
import argparse

import _bootstrap  # noqa: F401

from emotional_instability.capabilities import run_benchmark
from emotional_instability.config import load_all
from emotional_instability.eval import run_section2
from emotional_instability.petri import run_petri
from emotional_instability.prefill import run_recovery_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frustration", nargs="*", default=[],
                    help="models to re-run the Section 2 frustration eval on (Figure 5)")
    ap.add_argument("--petri", nargs="*", default=[], help="models for Petri elicitation (Figure 6)")
    ap.add_argument("--capabilities", nargs="*", default=[],
                    help="models for capability benchmarks (Figure 7)")
    ap.add_argument("--recovery", nargs="*", default=[],
                    help="models for the recovery probe (Section 4.2); needs --section2")
    ap.add_argument("--section2", default="artifacts/section2/gemma-3-27b-it.jsonl",
                    help="source of score>=7 responses for the recovery probe")
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    registry, cfg = load_all()
    if args.scale is not None:
        cfg.raw["scale"] = args.scale

    for model in args.frustration:
        run_section2(model, registry, cfg, out_dir="artifacts/section4/frustration")

    for model in args.petri:
        run_petri(model, registry, cfg)

    if args.recovery:
        run_recovery_probe(args.recovery, registry, cfg, section2_path=args.section2)

    if args.capabilities:
        benches = cfg.section("section4")["capabilities"]["benchmarks"]
        rows = []
        for model in args.capabilities:
            for b in benches:
                rows.append(run_benchmark(b, model, registry, cfg))
        print("\n=== Capability benchmarks (Figure 7) ===")
        for r in rows:
            print(f"{r['benchmark']:>12}  {r['model']:<22}  "
                  f"acc={r['accuracy'] if r['accuracy'] is None else round(r['accuracy'],3)}  n={r['n']}")


if __name__ == "__main__":
    main()
