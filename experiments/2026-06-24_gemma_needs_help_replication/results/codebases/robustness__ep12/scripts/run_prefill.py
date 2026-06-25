#!/usr/bin/env python
"""Section 3.1 base-vs-instruct prefill comparison (Gemma-only in this scope).

Steps (run in order; --build then --eval, or both):
    # 1. Build prefills from existing Gemma-27B-it elicitation results.
    python scripts/run_prefill.py --build \
        --source-results results/elicit_gemma27b.jsonl \
        --prefills runs/prefills.jsonl

    # 2. Generate + score continuations for a model (base or instruct).
    python scripts/run_prefill.py --eval --model gemma-3-27b-pt \
        --prefills runs/prefills.jsonl --out results/prefill_gemma27b_base.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config, prefill
from distress.judge import FrustrationJudge
from distress.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--source-results", help="Gemma-27B-it elicitation JSONL")
    ap.add_argument("--prefills", required=True)
    ap.add_argument("--model", help="target key (for --eval)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", help="continuation output (for --eval)")
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    models_cfg = config.load_models()
    exp = config.load_experiment()
    pf_cfg = exp["prefill"]

    if args.build:
        # Tokenizer + Claude helpers; use the instruct Gemma tokenizer.
        gemma = build_client(config.get_target("gemma-3-27b-it", models_cfg))
        _, tokenizer = gemma.get_model_and_tokenizer()
        labeller = build_client(config.get_judge("onset_labeller", models_cfg))
        paraphraser = build_client(config.get_judge("paraphraser", models_cfg))

        sources = prefill.select_high_frustration(
            args.source_results, n_numeric=pf_cfg["source_numeric"],
            n_text=pf_cfg["source_text"])
        prefills = prefill.build_prefills(
            sources, labeller, paraphraser, tokenizer,
            early_tokens=pf_cfg["early_truncate_tokens"],
            conditions=tuple(pf_cfg["conditions"]),
            do_paraphrase=not args.no_paraphrase)
        Path(args.prefills).parent.mkdir(parents=True, exist_ok=True)
        with open(args.prefills, "w") as fh:
            for pf in prefills:
                fh.write(json.dumps(pf.as_dict()) + "\n")
        print(f"[prefill] built {len(prefills)} prefills -> {args.prefills}")

    if args.eval:
        from distress.prefill import Prefill

        prefills = [Prefill(**json.loads(l))
                    for l in open(args.prefills) if l.strip()]
        target_entry = config.get_target(args.model, models_cfg)
        kwargs = {"adapter_path": args.adapter} if args.adapter else {}
        target = build_client(target_entry, **kwargs)
        judge = FrustrationJudge(
            build_client(config.get_judge("frustration_judge", models_cfg)))
        prefill.run_prefill_eval(
            target, judge, prefills, args.out,
            model_name=args.name or args.model,
            n_continuations=pf_cfg["continuations_per_prefill"],
            temperature=exp["sampling"]["temperature"])
        print(f"[prefill] wrote {args.out}")


if __name__ == "__main__":
    main()
