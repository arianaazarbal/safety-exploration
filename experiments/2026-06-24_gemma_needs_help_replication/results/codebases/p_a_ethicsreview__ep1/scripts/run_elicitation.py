#!/usr/bin/env python3
"""Section 2: run the elicitation suite for one or more target models.

Produces ~4000 judge-scored responses per model across the 8 conditions and
writes them to ``data/scores_<model>.jsonl``.

Example:
    python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_elicitation.py --models gemma-3-27b-it --conditions extended_8turn
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, make_judge, make_target, setup

from emotional_instability.eval.runner import run_elicitation_for_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Target model keys from config/models.yaml.")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Optional subset of condition names to run.")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter path (e.g. the DPO model). Applies "
                         "to a single Gemma target; output is tagged with the adapter.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load Gemma in 4-bit (fits the 27B model on a single GPU).")
    args = ap.parse_args()

    cfg = setup()
    judge = make_judge(cfg)

    for model_key in args.models:
        spec = cfg.target(model_key)
        kw = {"load_in_4bit": True} if (args.load_in_4bit and spec.backend == "gemma") else {}
        client = make_target(cfg, model_key, adapter_path=args.adapter, **kw)
        tag = "_dpo" if args.adapter else ""
        label = f"{model_key}{tag}"  # distinct 'model' field so vanilla/DPO don't collide
        out_path = DATA_DIR / f"scores_{label}.jsonl"
        run_elicitation_for_model(
            label, client, judge, cfg, out_path, conditions=args.conditions
        )
        print(f"[done] {label} -> {out_path}")


if __name__ == "__main__":
    main()
