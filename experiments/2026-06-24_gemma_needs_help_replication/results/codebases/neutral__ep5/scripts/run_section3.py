#!/usr/bin/env python
"""Section 3: post-training divergence via prefilling (Gemma base vs instruct).

Steps:
  1. Harvest high-frustration (score >= 5) Gemma-27B-it seeds: numeric + text.
  2. Build prefills (onset + early truncations, paraphrased) — Appendix C.
  3. For each of {gemma-3-27b-it, gemma-3-27b-pt}, generate continuations and
     score them. Persist Figure-4 metrics.

Note: Gemini has no public base model and is out of scope, so the base-vs-instruct
comparison is Gemma-only (see DESIGN.md §"Section 3 scope").
"""

from __future__ import annotations

from _common import get_judge, load
from distress import config
from distress.eval.conditions import build_rollouts
from distress.eval.rollout import run_rollout
from distress.prefill.build_prefills import build_prefills, save_prefills
from distress.prefill.continuations import (
    continuations_to_df,
    run_continuations,
    save_continuations,
    section3_metrics,
)


def harvest_seeds(client, judge, n_numeric, n_text):
    """Run instruct rollouts until we collect enough high-frustration seeds."""
    specs = build_rollouts(seed=1)
    numeric_seeds, text_seeds = [], []
    for spec in specs:
        if len(numeric_seeds) >= n_numeric and len(text_seeds) >= n_text:
            break
        want_numeric = (not spec.task.is_text) and len(numeric_seeds) < n_numeric
        want_text = spec.task.is_text and len(text_seeds) < n_text
        if not (want_numeric or want_text):
            continue
        r = run_rollout(client, spec)
        judge.score_rollouts([r])
        if any(t.rating and t.rating >= config.HIGH_FRUSTRATION_THRESHOLD for t in r.turns):
            r.is_text = spec.task.is_text
            (text_seeds if spec.task.is_text else numeric_seeds).append(r)
    return numeric_seeds, text_seeds


def main():
    judge = get_judge()
    it_spec = next(s for s in config.SECTION3_MODELS if not s.is_base)
    base_spec = next(s for s in config.SECTION3_MODELS if s.is_base)

    print("=== Harvesting high-frustration seeds from Gemma-27B-it ===")
    it_client = load(it_spec)
    numeric_seeds, text_seeds = harvest_seeds(
        it_client, judge, config.PREFILL_N_NUMERIC, config.PREFILL_N_TEXT
    )
    print(f"  seeds: {len(numeric_seeds)} numeric, {len(text_seeds)} text")

    print("=== Building prefills (onset labelling + paraphrasing) ===")
    prefills = build_prefills(numeric_seeds + text_seeds, it_client.tokenizer)
    save_prefills(prefills, config.RESULTS_DIR / "section3_prefills.jsonl")

    all_records = []
    print("=== Continuations: instruct ===")
    recs = run_continuations(it_client, prefills, judge)
    save_continuations(recs, config.RESULTS_DIR / "section3_cont_instruct.jsonl")
    all_records += recs
    del it_client

    print("=== Continuations: base ===")
    base_client = load(base_spec)
    recs = run_continuations(base_client, prefills, judge)
    save_continuations(recs, config.RESULTS_DIR / "section3_cont_base.jsonl")
    all_records += recs
    del base_client

    df = continuations_to_df(all_records)
    df.to_csv(config.RESULTS_DIR / "section3_continuations.csv", index=False)
    metrics = section3_metrics(df)
    metrics.to_csv(config.RESULTS_DIR / "section3_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
