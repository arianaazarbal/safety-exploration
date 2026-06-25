#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation.

Example:
    python scripts/06_run_petri.py --models Gemma-3-27B-it
    python scripts/06_run_petri.py --adapter checkpoints/dpo_Gemma-3-27B-it \
        --base Gemma-3-27B-it --label DPO-Gemma
"""

import _bootstrap  # noqa: F401
import argparse

from gemma_distress import config
from gemma_distress.petri import run_petri_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["Gemma-3-27B-it"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default="Gemma-3-27B-it")
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    args = ap.parse_args()

    by_name = {m.name: m for m in config.MAIN_EVAL_MODELS}

    if args.adapter:
        from gemma_distress.config import fine_tuned_spec
        base = by_name[args.base]
        spec = fine_tuned_spec(base, args.label or "DPO")
        out = run_petri_for_model(spec, adapter_path=args.adapter,
                                  n_per_emotion=args.n_per_emotion)
        print(f"[done] {out}")
        return

    for name in args.models:
        out = run_petri_for_model(by_name[name], n_per_emotion=args.n_per_emotion)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
