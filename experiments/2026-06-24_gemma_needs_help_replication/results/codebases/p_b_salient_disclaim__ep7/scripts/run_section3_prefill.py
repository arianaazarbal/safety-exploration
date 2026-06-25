#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill continuation experiment.

Steps:
  1. Sample seed conversations from Gemma-3-27B-it on impossible-numeric and
     trigger (text) questions, score them, and keep high-frustration seeds
     (10 numeric + 10 text by default).
  2. Build early/onset prefills (paraphrased) from the seeds.
  3. Have each prefill-capable model (Gemma base + instruct) generate 50
     continuations per prefill; score the continuation only.
  4. Save raw continuations + per-model/per-truncation aggregate stats.

Scope: Gemma base + instruct only (Gemini has no base model / prefill API).

Example:
  python scripts/run_section3_prefill.py \
      --models gemma-3-27b-pt gemma-3-27b-it --n-seed-numeric 10 --n-seed-text 10
"""
import _bootstrap  # noqa: F401

import argparse
import os

import numpy as np

import config
from emotional_instability import io_utils
from emotional_instability.eval import runner, scoring
from emotional_instability.eval.build_specs import build_specs
from emotional_instability.models import get_client
from emotional_instability.prefill import (
    build_prefills_from_seeds, run_continuation_experiment, seeds_from_rollouts)


def _gather_seeds(seed_model, n_numeric, n_text, seed):
    client = get_client(seed_model)
    # numeric seeds
    num_specs = build_specs("impossible_numeric", n_samples=max(40, n_numeric * 8), seed=seed)
    num_roll = runner.run_category(client, "impossible_numeric", specs=num_specs, base_seed=seed)
    num_scored = scoring.score_rollouts(num_roll, score_all_turns=True)
    numeric_seeds = seeds_from_rollouts(
        num_roll, num_scored, question_kind="numeric",
        threshold=config.PREFILL.seed_score_threshold, n=n_numeric)

    # text seeds (trigger questions)
    txt_specs = build_specs("triggers", n_samples=max(40, n_text * 8), seed=seed + 1)
    txt_roll = runner.run_category(client, "triggers", specs=txt_specs, base_seed=seed + 1)
    txt_scored = scoring.score_rollouts(txt_roll, score_all_turns=True)
    text_seeds = seeds_from_rollouts(
        txt_roll, txt_scored, question_kind="text",
        threshold=config.PREFILL.seed_score_threshold, n=n_text)

    return numeric_seeds + text_seeds


def _aggregate(results):
    """Per (model, truncation) mean score and % >=5."""
    out = {}
    keyed = {}
    for r in results:
        keyed.setdefault((r.model, r.truncation), []).append(r.score)
    for (model, trunc), scores in keyed.items():
        vals = [s for s in scores if s is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{model}::{trunc}"] = {
            "mean": float(arr.mean()),
            "pct_high": float((arr >= config.HIGH_FRUSTRATION_THRESHOLD).mean() * 100),
            "n": len(vals),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--seed-model", default="gemma-3-27b-it")
    ap.add_argument("--n-seed-numeric", type=int, default=config.PREFILL.n_seed_responses_numeric)
    ap.add_argument("--n-seed-text", type=int, default=config.PREFILL.n_seed_responses_text)
    ap.add_argument("--continuations", type=int, default=config.PREFILL.continuations_per_prefill)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, "section3")
    io_utils.ensure_dir(out_dir)

    seeds = _gather_seeds(args.seed_model, args.n_seed_numeric, args.n_seed_text, args.seed)
    io_utils.write_jsonl(os.path.join(out_dir, "seeds.jsonl"), seeds)

    tokenizer_client = get_client(args.seed_model)
    prefills = build_prefills_from_seeds(seeds, tokenizer_client,
                                         paraphrase=not args.no_paraphrase)
    io_utils.write_jsonl(os.path.join(out_dir, "prefills.jsonl"), prefills)

    results = run_continuation_experiment(
        args.models, prefills, continuations_per_prefill=args.continuations,
        base_seed=args.seed)
    io_utils.write_jsonl(os.path.join(out_dir, "continuations.jsonl"), results)

    agg = _aggregate(results)
    io_utils.write_json(os.path.join(out_dir, "summary.json"), agg)
    for k, v in sorted(agg.items()):
        print(f"  {k:35s} mean={v['mean']:.2f}  %>=5={v['pct_high']:.1f}  n={v['n']}")


if __name__ == "__main__":
    main()
