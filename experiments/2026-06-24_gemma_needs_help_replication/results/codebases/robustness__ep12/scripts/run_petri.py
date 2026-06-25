#!/usr/bin/env python
"""Open-ended Petri-style emotion elicitation (Section 4 / Appendix G).

    python scripts/run_petri.py --model gemma-3-27b-it \
        --out results/petri_gemma27b.jsonl
    python scripts/run_petri.py --model gemma-3-27b-it --adapter runs/dpo \
        --name dpo-gemma --out results/petri_dpo.jsonl
    python scripts/run_petri.py --summarise results/petri_*.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config, petri_eval
from distress.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="target key")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarise", nargs="*", default=None)
    args = ap.parse_args()

    if args.summarise:
        import pandas as pd
        frames = [petri_eval.summarise_petri(p) for p in args.summarise]
        print(pd.concat(frames).to_string(index=False))
        return

    models_cfg = config.load_models()
    exp = config.load_experiment()
    pc = exp["petri"]

    kwargs = {"adapter_path": args.adapter} if args.adapter else {}
    target = build_client(config.get_target(args.model, models_cfg), **kwargs)
    auditor = build_client(config.get_judge("petri_auditor", models_cfg))
    judge = build_client(config.get_judge("petri_judge", models_cfg))

    petri_eval.run_petri(
        auditor, target, judge, args.out,
        model_name=args.name or args.model,
        emotions=tuple(pc["emotions"]),
        transcripts_per_emotion=pc["transcripts_per_emotion"],
        max_turns=pc["max_turns"])
    print(f"[petri] wrote {args.out}")


if __name__ == "__main__":
    main()
