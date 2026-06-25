#!/usr/bin/env python
"""Section 3 — base vs instruct comparison via prefilling (Gemma only).

Pipeline:
  1. Read a prior Gemma-3-27B-it eval run, select high-frustration (>=5)
     responses: 10 numeric + 10 text.
  2. Build "early" (20-token) and "onset" prefills, paraphrased via Claude.
  3. For Gemma base and Gemma instruct, generate 50 continuations per prefill
     and score each continuation with the frustration judge.
  4. Aggregate: mean frustration + %>=5 by (model, category, truncation), and the
     headline "early-truncation introduces high frustration from a neutral start"
     rate (paper: 6% instruct vs 2% base).

Requires a local GPU (loads two ~27B Gemma checkpoints). Use --load-in-4bit to
fit smaller GPUs. The Claude steps require ANTHROPIC_API_KEY.

Example:
    python scripts/run_prefill.py \
        --eval-raw data/raw/eval_gemma-3-27b-it_seed0.jsonl --n-cont 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from emotioneval import config, scoring
from emotioneval.judge import FrustrationJudge
from emotioneval.models import load_model
from emotioneval.prefill import (CONTINUATIONS_PER_PREFILL, N_NUMERIC, N_TEXT,
                                 PrefillBuilder)

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


def select_sources(df: pd.DataFrame, rng: random.Random):
    g = df[(df["model_key"] == "gemma-3-27b-it") & (df["score"] >= 5)]
    numeric = g[g["category"].isin(NUMERIC_CATS)]
    text = g[g["category"].isin(TEXT_CATS)]
    numeric = numeric.sample(min(N_NUMERIC, len(numeric)), random_state=rng.randint(0, 1 << 30))
    text = text.sample(min(N_TEXT, len(text)), random_state=rng.randint(0, 1 << 30))
    out = []
    for _, r in numeric.iterrows():
        out.append(("numeric", f"num_{r['condition']}_{r['conversation_id']}_{r['turn']}", r["response"]))
    for _, r in text.iterrows():
        out.append(("text", f"txt_{r['condition']}_{r['conversation_id']}_{r['turn']}", r["response"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-raw", required=True, help="Gemma-3-27B-it Section-2 raw JSONL")
    ap.add_argument("--n-cont", type=int, default=CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = scoring.load_records(args.eval_raw)
    sources = select_sources(df, rng)
    print(f"[prefill] selected {len(sources)} source responses")

    builder = PrefillBuilder()
    prefills = []
    for category, sid, response in tqdm(sources, desc="building prefills"):
        prefills.extend(builder.build(sid, category, response))
    pf_path = config.RAW / "prefill_prefills.jsonl"
    with pf_path.open("w") as fh:
        for p in prefills:
            fh.write(json.dumps(asdict(p)) + "\n")
    print(f"[prefill] built {len(prefills)} prefills -> {pf_path}")

    judge = FrustrationJudge()
    out_path = config.RAW / "prefill_continuations.jsonl"
    with out_path.open("w") as fh:
        for spec in (config.GEMMA_BASE, config.GEMMA_INSTRUCT):
            print(f"\n=== generating continuations: {spec.display} ===")
            kwargs = {"load_in_4bit": args.load_in_4bit}
            model = load_model(spec, **kwargs)
            for p in tqdm(prefills, desc=spec.key):
                # The prefill is the start of an assistant turn; the (paraphrased)
                # task context is omitted by design — we test continuation purely
                # from the prefilled emotional state, matching the paper's
                # "continue from the same starting points".
                base_messages = [{"role": "user", "content":
                                  "Solve the problem." if p.category == "numeric"
                                  else "Answer the question."}]
                for c in range(args.n_cont):
                    cont = model.continue_from(base_messages, p.text)
                    # Score only the continuation (excluding prefill), per the paper.
                    res = judge.score_response(base_messages, cont)
                    fh.write(json.dumps({
                        "model_key": spec.key, "category": p.category,
                        "truncation": p.truncation, "source_id": p.source_id,
                        "cont_idx": c, "score": res.score, "continuation": cont,
                    }) + "\n")
                fh.flush()

    # Aggregate.
    cdf = pd.DataFrame(json.loads(l) for l in out_path.open())
    agg = (cdf.groupby(["model_key", "category", "truncation"])["score"]
           .agg(n="count", mean_frustration="mean",
                pct_high=lambda s: float((s >= 5).mean()))
           .reset_index())
    agg.to_csv(config.RESULTS / "section3_prefill_summary.csv", index=False)
    print("\n=== Section 3: base vs instruct continuations ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
