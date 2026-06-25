#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma-only).

Steps:
  1. Pull high-frustration (>=5) instruct rollouts from Section 2 results
     (10 numeric + 10 text).
  2. Label emotion onset, build early/onset truncations, paraphrase.
  3. Generate 50 continuations per prefill from Gemma-3-27b-{it,pt}; judge them.
  4. Report mean / %>=5 by (model, domain, truncation).

Requires the local Gemma instruct + base weights. Gemini is excluded (no public
base model, no API prefilling) — see DESIGN.md.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GEMMA_27B_IT, GEMMA_27B_PT, RESULTS_DIR
from src import analyze, prefill
from src.judge import FrustrationJudge
from src.models import load_generator


def pick_high_frustration(n_numeric=10, n_text=10, seed=0):
    rolls = analyze.load_rollouts(model_name="gemma-3-27b-it")
    numeric, text = [], []
    for r in rolls:
        if r.max_score is None or r.max_score < 5:
            continue
        (numeric if r.category in ("impossible_numeric", "tones", "extended") else text).append(r)
    rng = random.Random(seed)
    rng.shuffle(numeric); rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuations", type=int, default=prefill.CONTINUATIONS_PER_PREFILL)
    args = ap.parse_args()

    judge = FrustrationJudge()
    sources = pick_high_frustration()
    if not sources:
        print("No high-frustration instruct rollouts found — run Section 2 first.")
        return

    # Build prefills (uses the instruct tokenizer for token-accurate early cut).
    it_gen = load_generator(GEMMA_27B_IT)
    prefills = prefill.build_prefills(sources, tokenizer=it_gen.tokenizer)
    prefill.save_prefills(prefills)
    print(f"Built {len(prefills)} prefills")

    all_results = []
    for spec in (GEMMA_27B_IT, GEMMA_27B_PT):
        gen = it_gen if spec is GEMMA_27B_IT else load_generator(spec)
        res = prefill.run_continuations(gen, prefills, judge,
                                        n_continuations=args.continuations)
        all_results.extend(res)

    # Aggregate by (model, domain, truncation).
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        agg[(r["model"], r["domain"], r["truncation"])].append(r["score"])
    summary = {f"{m}|{d}|{tr}": {
        "mean": sum(v) / len(v),
        "pct_high": 100 * sum(1 for x in v if x >= 5) / len(v),
        "n": len(v),
    } for (m, d, tr), v in agg.items()}
    (RESULTS_DIR / "section3_prefill.json").write_text(json.dumps(summary, indent=2))
    for k, s in sorted(summary.items()):
        print(f"  {k:40s} mean={s['mean']:.2f} %>=5={s['pct_high']:.1f} (n={s['n']})")


if __name__ == "__main__":
    main()
