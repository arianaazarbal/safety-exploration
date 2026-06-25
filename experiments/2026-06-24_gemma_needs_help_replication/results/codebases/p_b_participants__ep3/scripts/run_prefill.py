#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill comparison (Gemma only).

Pipeline (paper §3.1):
  1. Load §2 scored responses for Gemma-3-27B-it; sample 20 high-frustration
     (score>=5) source responses: 10 numeric, 10 text.
  2. For each, label the emotional onset (Claude), then truncate "early"
     (20 tokens) and "onset" (numeric uses both; text uses onset only).
  3. Paraphrase each truncation (Claude) to neutralise Gemma's surface style.
  4. For Gemma base and Gemma instruct, generate 50 continuations per prefill,
     score the continuation only, and aggregate by (model, kind, category).

Scope: Gemini and the other families are out (no base checkpoint / no prefill).

Example:
    python scripts/run_prefill.py --results artifacts/eval/gemma-3-27b-it.jsonl \
        --base gemma-3-27b-pt --instruct gemma-3-27b-it --out artifacts/prefill
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from emotional_instability.config import ModelsConfig
from emotional_instability.prefill import (
    PrefillItem,
    label_onset,
    paraphrase_prefix,
    run_prefill_continuations,
    truncate_at_onset,
    truncate_early,
)
from emotional_instability.prefill.continuation import aggregate_continuations
from emotional_instability.runtime import (
    get_participant,
    get_prefill_helper,
    get_judge,
    setup_logging,
)
from emotional_instability.scoring import FrustrationScorer
from emotional_instability.storage import load_results_jsonl, save_json


def _sample_sources(results, n_each, seed):
    rng = random.Random(seed)
    numeric = [r for r in results if r.category == "impossible_numeric" and (r.score or 0) >= 5]
    text = [r for r in results if r.category in ("triggers",) and (r.score or 0) >= 5]
    return rng.sample(numeric, min(n_each, len(numeric))), rng.sample(text, min(n_each, len(text)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="Gemma-3-27B-it §2 scored JSONL")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--n-each", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--early-tokens", type=int, default=20)
    ap.add_argument("--out", default="artifacts/prefill")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    helper = get_prefill_helper(models_cfg)
    scorer = FrustrationScorer(get_judge(models_cfg, "frustration"))
    out_dir = Path(args.out)

    results = load_results_jsonl(args.results)
    numeric, text = _sample_sources(results, args.n_each, args.seed)

    # We need the instruct tokenizer for token-accurate truncation; build it once.
    instruct = get_participant(models_cfg, args.instruct)
    tokenizer = instruct.tokenizer() if hasattr(instruct, "tokenizer") else None

    items: list[PrefillItem] = []
    for sid, r in enumerate(numeric):
        onset = label_onset(r.response, helper)
        for trunc in (
            truncate_early(r.response, n_tokens=args.early_tokens, tokenizer=tokenizer),
            truncate_at_onset(r.response, onset, tokenizer=tokenizer),
        ):
            items.append(PrefillItem(
                source_id=sid, question=r.seed_prompt, category="numeric",
                kind=trunc.kind, prefix=paraphrase_prefix(trunc.text, helper),
                source_score=r.score,
            ))
    for sid, r in enumerate(text, start=1000):
        # Text: only the "onset" truncation (early yields minimal emotion).
        onset = label_onset(r.response, helper)
        trunc = truncate_at_onset(r.response, onset, tokenizer=tokenizer)
        items.append(PrefillItem(
            source_id=sid, question=r.seed_prompt, category="text",
            kind="onset", prefix=paraphrase_prefix(trunc.text, helper),
            source_score=r.score,
        ))

    save_json([item.__dict__ for item in items], out_dir / "prefill_items.json")

    all_aggs = []
    for model_name in (args.base, args.instruct):
        model = (
            instruct if model_name == args.instruct
            else get_participant(models_cfg, model_name)
        )
        conts = run_prefill_continuations(
            model, items, scorer, n_continuations=args.n_continuations,
        )
        aggs = aggregate_continuations(conts)
        all_aggs.extend(a.__dict__ for a in aggs)
        print(f"\n===== {model_name} continuation frustration =====")
        for a in aggs:
            print(f"  {a.category:8s} {a.kind:6s}  mean={a.mean_score:.2f}  %>=5={a.pct_high:.1f}  (n={a.n})")
        if model_name != args.instruct:
            model.close()

    save_json(all_aggs, out_dir / "prefill_aggregates.json")
    print(f"\nSaved prefill results under {out_dir}/")


if __name__ == "__main__":
    main()
