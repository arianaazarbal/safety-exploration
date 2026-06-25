#!/usr/bin/env python
"""Section 4 evaluation: compare vanilla / DPO / SFT Gemma.

Runs the Section 2 distress suite, the Petri open-ended elicitation, and the
capability benchmarks on the finetuned models, then renders Figures 5-7.

Usage:
    python -m scripts.run_section4_eval --dpo outputs/checkpoints/dpo \
        --sft outputs/checkpoints/sft
    python -m scripts.run_section4_eval --dpo ... --petri --capabilities
"""
from __future__ import annotations

import argparse

import config
from emotional_instability.eval import runner
from emotional_instability.eval import metrics as M
from emotional_instability.analysis import figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", default=None, help="path to DPO LoRA adapter")
    ap.add_argument("--sft", default=None, help="path to SFT LoRA adapter")
    ap.add_argument("--petri", action="store_true")
    ap.add_argument("--capabilities", action="store_true")
    ap.add_argument("--cap-limit", type=int, default=None)
    args = ap.parse_args()

    variants = {"gemma-vanilla": None}
    if args.dpo:
        variants["gemma-dpo"] = args.dpo
    if args.sft:
        variants["gemma-sft"] = args.sft

    # --- Section 2 distress suite on each variant ---
    for key, adapter in variants.items():
        print(f"[gen] {key}")
        resp = runner.generate_adapter_model(key, adapter)
        print(f"[score] {key}")
        runner.score_file(resp)

    df = M.load_all()
    sub = df[df["model"].isin(variants)]
    if not sub.empty:
        print(M.figure1_table(sub).to_string(index=False))
        print("wrote", figures.figure5(df, list(variants)))

    # --- Petri ---
    if args.petri:
        from emotional_instability.petri.run_petri import run_petri
        from emotional_instability.petri import metrics as PMet
        paths = []
        paths.append(run_petri("gemma-3-27b-it"))
        if args.dpo:
            paths.append(run_petri("gemma-3-27b-it", adapter_path=args.dpo))
        pdf = PMet.load(paths)
        print(PMet.figure6_table(pdf).to_string(index=False))
        print("wrote", figures.figure6(pdf))

    # --- Capabilities ---
    if args.capabilities:
        from emotional_instability.capabilities import benchmarks as C
        base = config.FINETUNE_BASE.model_id
        vanilla = C.run_lm_eval(base, adapter_path=None, limit=args.cap_limit)
        if args.dpo:
            ft = C.run_lm_eval(base, adapter_path=args.dpo, limit=args.cap_limit)
            diff = C.diff_table(vanilla, ft)
            print("Capability deltas (DPO - vanilla):")
            for task, vals in diff.items():
                print(f"  {task}: {vals}")
            print("wrote", figures.figure7(diff))


if __name__ == "__main__":
    main()
