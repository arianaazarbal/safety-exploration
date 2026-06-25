#!/usr/bin/env python
"""Section 4.2: recovery limitation (Figure 8).

Tests whether a model can climb OUT of a frustration spiral, rather than just
avoid entering one. We take extremely high-frustration responses (score>=7),
truncate them 200 tokens before their end, paraphrase, and measure continuations.
The paper finds 38% of DPO-model continuations still score>=5 — comparable to the
base model: no model reliably recovers from a highly-negative prefilled state.

Run for the vanilla and DPO Gemma (and optionally base) and compare %>=5.

Example:
    python scripts/run_recovery.py --results artifacts/eval/gemma-3-27b-it.jsonl \
        --models gemma-3-27b-it --adapters none artifacts/training/dpo --out artifacts/recovery
"""
from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.config import ModelsConfig
from emotional_instability.prefill import (
    PrefillItem,
    paraphrase_prefix,
    run_prefill_continuations,
    truncate_before_end,
)
from emotional_instability.prefill.continuation import aggregate_continuations
from emotional_instability.runtime import (
    get_judge,
    get_participant,
    get_prefill_helper,
    setup_logging,
)
from emotional_instability.scoring import FrustrationScorer
from emotional_instability.storage import load_results_jsonl, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="§2 scored JSONL with high-frustration responses")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapters", nargs="+", default=["none"],
                    help="adapter dirs to compare; 'none' = vanilla")
    ap.add_argument("--n-sources", type=int, default=20)
    ap.add_argument("--tokens-from-end", type=int, default=200)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--out", default="artifacts/recovery")
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    helper = get_prefill_helper(models_cfg)
    scorer = FrustrationScorer(get_judge(models_cfg, "frustration"))

    results = load_results_jsonl(args.results)
    extreme = [r for r in results if (r.score or 0) >= 7][: args.n_sources]
    if not extreme:
        raise SystemExit("No score>=7 responses found in --results; cannot test recovery.")

    base_model = get_participant(models_cfg, args.model)
    tokenizer = base_model.tokenizer() if hasattr(base_model, "tokenizer") else None

    items = []
    for sid, r in enumerate(extreme):
        trunc = truncate_before_end(r.response, n_tokens_from_end=args.tokens_from_end, tokenizer=tokenizer)
        items.append(PrefillItem(
            source_id=sid, question=r.seed_prompt, category="recovery",
            kind="recovery", prefix=paraphrase_prefix(trunc.text, helper), source_score=r.score,
        ))

    summary = []
    for adapter in args.adapters:
        adapter_path = None if adapter == "none" else adapter
        model = (
            base_model if adapter == "none"
            else get_participant(models_cfg, args.model, adapter_path=adapter_path)
        )
        conts = run_prefill_continuations(model, items, scorer, n_continuations=args.n_continuations)
        agg = aggregate_continuations(conts)[0]
        summary.append({"adapter": adapter, "mean": agg.mean_score, "pct_high": agg.pct_high, "n": agg.n})
        print(f"  {adapter:30s}: mean={agg.mean_score:.2f}  %>=5={agg.pct_high:.1f}  (n={agg.n})")
        if adapter != "none":
            model.close()
    base_model.close()

    save_json(summary, Path(args.out) / "recovery_summary.json")


if __name__ == "__main__":
    main()
