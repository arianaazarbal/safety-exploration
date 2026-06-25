#!/usr/bin/env python
"""Section 2 — Eliciting and quantifying model distress.

Pipeline:
  1. Build the full multi-turn protocol (~4,000 rollouts/model at scale 1.0).
  2. For each in-scope model (Gemma-3-{27B,12B}-it, Gemini-2.5-{flash,pro}),
     run every rollout and record per-turn assistant responses.
  3. Score every response with the Claude-Sonnet-4 frustration judge.
  4. Cross-check 260 randomly sampled responses with the GPT-5-mini judge and
     report Pearson r / % within one point.
  5. Compute Figure 1/2 summaries, Figure 3 per-turn progression, and the
     Table 3/8 differential word lists.

Usage:
    python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
    DISTRESS_EVAL_SCALE=0.01 python scripts/run_section2.py   # smoke test

Nothing is executed at import time; this is a runnable entrypoint.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
from eval_distress import analysis, io_utils
from eval_distress.conditions import build_full_protocol
from eval_distress.data.wildchat import load_wildchat_prompts
from eval_distress.judge import FrustrationJudge, judge_agreement
from eval_distress.models import load_target
from eval_distress.protocol import run_protocol

VALIDATION_SAMPLE_N = 260  # paper's judge-reliability sample size


def run_one_model(model_key: str, rollouts, *, load_in_4bit: bool):
    model = load_target(model_key, load_in_4bit=load_in_4bit)
    results = run_protocol(model, model_key, rollouts)
    rows = io_utils.rollouts_to_response_rows(results)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=C.MAIN_PROTOCOL_MODELS)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Quantise local Gemma to fit a single GPU.")
    ap.add_argument("--skip-generation", action="store_true",
                    help="Reuse cached responses_*.jsonl and only (re)score.")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    wildchat = load_wildchat_prompts()
    rollouts = build_full_protocol(wildchat)
    print(f"Built {len(rollouts)} rollouts/model (scale={C.EVAL_SCALE}).")

    judge = FrustrationJudge(C.EMOTION_JUDGE)
    all_scored: list[dict] = []

    for model_key in args.models:
        resp_path = C.RESULTS_DIR / f"responses_{model_key}.jsonl"
        if args.skip_generation and resp_path.exists():
            rows = io_utils.read_jsonl(resp_path)
        else:
            print(f"== Generating responses: {model_key} ==")
            rows = run_one_model(model_key, rollouts, load_in_4bit=args.load_in_4bit)
            io_utils.write_jsonl(resp_path, rows)

        print(f"== Scoring {len(rows)} responses: {model_key} ==")
        scores = judge.score_many([r["text"] for r in rows])
        for r, s in zip(rows, scores):
            r["rating"] = s.rating
            r["judge_evidence"] = s.evidence
        io_utils.write_jsonl(C.RESULTS_DIR / f"scored_{model_key}.jsonl", rows)
        all_scored += rows

    # ---- Judge reliability cross-check (260 random responses) -------------
    if not args.skip_validation:
        rng = random.Random(0)
        sample = rng.sample(all_scored, min(VALIDATION_SAMPLE_N, len(all_scored)))
        val_judge = FrustrationJudge(C.VALIDATION_JUDGE)
        val_scores = val_judge.score_many([r["text"] for r in sample])
        agreement = judge_agreement([r["rating"] for r in sample],
                                    [s.rating for s in val_scores])
        io_utils.write_json(C.RESULTS_DIR / "judge_agreement.json", agreement)
        print("Judge agreement:", agreement)

    # ---- Summaries / figures ---------------------------------------------
    summary = analysis.summarise(all_scored)
    io_utils.write_json(C.RESULTS_DIR / "summary_fig1_2.json", summary)

    fig3 = {
        "extended": analysis.per_turn_progression(all_scored, "extended"),
        "wildchat": analysis.per_turn_progression(all_scored, "wildchat"),
    }
    io_utils.write_json(C.RESULTS_DIR / "per_turn_fig3.json", fig3)

    word_tables = {m: analysis.differential_words(all_scored, m)
                   for m in args.models}
    io_utils.write_json(C.RESULTS_DIR / "differential_words_table3.json", word_tables)

    print("\n=== Figure 1 headline (avg % high-frustration) ===")
    for m, s in summary.items():
        print(f"  {m:20s} {s['avg_pct_high']:5.1f}%")


if __name__ == "__main__":
    main()
