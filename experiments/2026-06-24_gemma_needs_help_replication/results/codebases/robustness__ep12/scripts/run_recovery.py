#!/usr/bin/env python
"""Recovery-from-frustration experiment (Section 4.2). Gemma-only.

    python scripts/run_recovery.py --build \
        --source-results results/elicit_gemma27b.jsonl \
        --prefills runs/recovery_prefills.jsonl
    python scripts/run_recovery.py --eval --model gemma-3-27b-it \
        --adapter runs/dpo --name dpo-gemma \
        --prefills runs/recovery_prefills.jsonl --out results/recovery_dpo.jsonl
    python scripts/run_recovery.py --summarise results/recovery_*.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config, recovery
from distress.judge import FrustrationJudge
from distress.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--summarise", nargs="*", default=None)
    ap.add_argument("--source-results")
    ap.add_argument("--prefills")
    ap.add_argument("--model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.summarise:
        import pandas as pd
        frames = [recovery.summarise_recovery(p) for p in args.summarise]
        print(pd.concat(frames).to_string(index=False))
        return

    models_cfg = config.load_models()
    exp = config.load_experiment()

    if args.build:
        gemma = build_client(config.get_target("gemma-3-27b-it", models_cfg))
        _, tok = gemma.get_model_and_tokenizer()
        paraphraser = build_client(config.get_judge("paraphraser", models_cfg))
        prefills = recovery.build_recovery_prefills(
            args.source_results, tok, paraphraser)
        Path(args.prefills).parent.mkdir(parents=True, exist_ok=True)
        with open(args.prefills, "w") as fh:
            for pf in prefills:
                fh.write(json.dumps(pf) + "\n")
        print(f"[recovery] built {len(prefills)} prefills -> {args.prefills}")

    if args.eval:
        prefills = [json.loads(l) for l in open(args.prefills) if l.strip()]
        kwargs = {"adapter_path": args.adapter} if args.adapter else {}
        target = build_client(config.get_target(args.model, models_cfg),
                              **kwargs)
        judge = FrustrationJudge(
            build_client(config.get_judge("frustration_judge", models_cfg)))
        recovery.run_recovery(target, judge, prefills, args.out,
                              model_name=args.name or args.model,
                              temperature=exp["sampling"]["temperature"])
        print(f"[recovery] wrote {args.out}")


if __name__ == "__main__":
    main()
