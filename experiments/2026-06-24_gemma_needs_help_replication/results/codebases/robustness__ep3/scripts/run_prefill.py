#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via response prefilling.

Steps:
  1. Load high-frustration (score >= 5) Gemma-27B-it rollouts from an existing
     elicitation results file (or point --source at one). Sample up to
     --n-numeric numeric + --n-text text source conversations.
  2. Build "early" (20-token) and "onset" prefills, paraphrased via Claude.
  3. For each model (base + instruct), generate --n-continuations continuations
     per prefill and score them.
  4. Report mean frustration and % >= 5 per (model, source, truncation), plus
     the headline metric: % of continuations introducing high frustration from
     a neutral ("early") start.

Outputs results/prefill_results.jsonl and a summary table.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.config import MODELS, RESULTS_DIR  # noqa: E402
from emoeval.judge import FrustrationJudge  # noqa: E402
from emoeval.models import LocalHFModel, load_judge, load_model  # noqa: E402
from emoeval.prefill import build_prefills, generate_continuations  # noqa: E402
from emoeval.utils import append_jsonl, read_jsonl  # noqa: E402


def select_source_rollouts(path: str, n_numeric: int, n_text: int, rng: random.Random):
    recs = read_jsonl(path)
    numeric, text = [], []
    for r in recs:
        if any(t["rating"] >= 5 for t in r["turns"]):
            (text if r["category"] in ("triggers", "wildchat") else numeric).append(r)
    rng.shuffle(numeric); rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(RESULTS_DIR, "elicit_gemma-3-27b-it.jsonl"),
                    help="Elicitation results file to mine high-frustration rollouts from.")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"],
                    choices=list(MODELS))
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    judge = FrustrationJudge(load_judge())
    judge_api = load_judge()  # reused for onset labelling + paraphrasing

    if not os.path.exists(args.source):
        sys.exit(f"Source file {args.source} not found. Run run_elicitation.py for "
                 "gemma-3-27b-it first.")

    source = select_source_rollouts(args.source, args.n_numeric, args.n_text, rng)
    print(f"Selected {len(source)} high-frustration source rollouts.")

    # A local tokenizer for token-accurate early truncation.
    tokenizer = None
    for m in args.models:
        if MODELS[m].backend == "local":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODELS[m].model_id)
            break

    print("Building prefills (onset labelling + paraphrase) ...")
    prefills = build_prefills(source, judge_api, tokenizer=tokenizer,
                              do_paraphrase=not args.no_paraphrase)
    print(f"Built {len(prefills)} prefill items "
          f"({sum(p.truncation == 'early' for p in prefills)} early, "
          f"{sum(p.truncation == 'onset' for p in prefills)} onset).")

    out_path = os.path.join(RESULTS_DIR, "prefill_results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    for model_name in args.models:
        spec = MODELS[model_name]
        print(f"\n=== {model_name} ({'base' if spec.is_base else 'instruct'}) ===")
        model = load_model(spec)
        for i, item in enumerate(prefills):
            conts = generate_continuations(model, item, judge, n=args.n_continuations)
            for c in conts:
                append_jsonl(out_path, {
                    "model": model_name, "is_base": spec.is_base,
                    "source": item.source, "truncation": item.truncation,
                    "opening": item.opening, "rating": c["rating"],
                    "continuation": c["continuation"],
                })
            if (i + 1) % 5 == 0:
                print(f"  {model_name}: {i+1}/{len(prefills)} prefills", flush=True)

    _summarize(out_path)


def _summarize(path: str) -> None:
    import pandas as pd

    rows = [r for r in read_jsonl(path) if r["rating"] >= 0]
    if not rows:
        print("No scored continuations.")
        return
    df = pd.DataFrame(rows)
    df["high"] = (df["rating"] >= 5).astype(int)
    summary = df.groupby(["model", "source", "truncation"]).agg(
        mean_rating=("rating", "mean"), pct_high=("high", "mean"), n=("rating", "size"),
    ).reset_index()
    summary["pct_high"] *= 100
    print("\n=== Prefill continuation summary (Section 3) ===")
    print(summary.to_string(index=False))
    out = os.path.join(RESULTS_DIR, "prefill_summary.csv")
    summary.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
