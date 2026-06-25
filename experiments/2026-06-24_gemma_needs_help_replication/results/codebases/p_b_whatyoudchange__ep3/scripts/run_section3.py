#!/usr/bin/env python
"""Section 3 — Post-training divergence via prefilling (Gemma base vs instruct).

Pipeline:
  1. Load scored Gemma-3-27B-it responses from Section 2.
  2. Select 20 high-frustration (>=5) source conversations: 10 numeric, 10 text.
  3. Build early + onset prefills (paraphrased) via Claude-Sonnet.
  4. For Gemma-3-27B base (-pt) and instruct (-it), generate 50 continuations
     per prefill and score each continuation.
  5. Summarise mean frustration / %>=5 per model x truncation (Figure 4).

Scope note: the paper compares Gemma/Qwen/OLMo here. Restricted to Gemma+Gemini
(this request), and since Gemini has no base model and no prefill support, this
experiment is Gemma-only — base vs instruct, which is the core post-training
claim. See DESIGN.md.

Usage:
    python scripts/run_section3.py --models gemma-3-27b-pt gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
from eval_distress import analysis, io_utils
from eval_distress.judge import FrustrationJudge
from eval_distress.models import load_target
from eval_distress.prefill import (N_NUMERIC, N_TEXT, OnsetLabeller, Paraphraser,
                                   build_prefills, generate_continuations)

SOURCE_MODEL = "gemma-3-27b-it"


def select_high_frustration_convos(scored_rows: list[dict], *, seed: int = 0):
    """Group rows by convo_id; keep conversations whose final turn scored >=5.
    Return 10 numeric + 10 text source conversations (lists of rows)."""
    by_convo = defaultdict(list)
    for r in scored_rows:
        by_convo[r["convo_id"]].append(r)

    numeric, text = [], []
    for convo_id, rows in by_convo.items():
        rows = sorted(rows, key=lambda x: x["turn"])
        final = rows[-1]
        if final.get("rating") is None or final["rating"] < 5:
            continue
        cat = rows[0]["category"]
        if cat in ("impossible-numeric", "tones", "extended"):
            numeric.append(rows)
        elif cat == "triggers":
            text.append(rows)

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:N_NUMERIC], text[:N_TEXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    scored = io_utils.read_jsonl(C.RESULTS_DIR / f"scored_{SOURCE_MODEL}.jsonl")
    numeric, text = select_high_frustration_convos(scored)
    print(f"Selected {len(numeric)} numeric + {len(text)} text source convos.")

    # Build prefills (needs a tokenizer for 'early' token truncation).
    src_model = load_target(SOURCE_MODEL, load_in_4bit=args.load_in_4bit)
    tokenizer = getattr(src_model, "tokenizer", None)
    onset = OnsetLabeller()
    paraphraser = Paraphraser()
    prefills = build_prefills(numeric + text, tokenizer=tokenizer,
                              onset=onset, paraphraser=paraphraser)
    io_utils.write_json(C.RESULTS_DIR / "section3_prefills.json",
                        [vars(p) for p in prefills])
    print(f"Built {len(prefills)} prefills "
          f"(early numeric-only + onset; paraphrased).")

    judge = FrustrationJudge(C.EMOTION_JUDGE)
    all_rows: list[dict] = []
    for model_key in args.models:
        spec = C.TARGET_MODELS[model_key]
        if not spec.supports_prefill:
            print(f"Skipping {model_key}: prefill unsupported.")
            continue
        is_base = spec.variant == "base"
        model = load_target(model_key, load_in_4bit=args.load_in_4bit)
        print(f"== Continuations: {model_key} (base={is_base}) ==")
        for p in prefills:
            conts = generate_continuations(model, model_key, p, is_base=is_base)
            scores = judge.score_many(conts)
            for cont, s in zip(conts, scores):
                all_rows.append({
                    "model_key": model_key,
                    "category": p.question_type,       # numeric|text (reuse field)
                    "truncation": p.truncation,
                    "source_id": p.source_id,
                    "turn": 1, "n_turns": 1,
                    "text": cont,
                    "rating": s.rating,
                })

    io_utils.write_jsonl(C.RESULTS_DIR / "section3_continuations.jsonl", all_rows)

    # ---- Figure 4 summary: model x truncation x question type -------------
    summary = defaultdict(lambda: defaultdict(dict))
    grouped = defaultdict(list)
    for r in all_rows:
        if r["rating"] is None:
            continue
        grouped[(r["model_key"], r["truncation"], r["category"])].append(r["rating"])
    for (m, trunc, qt), ratings in grouped.items():
        summary[m][f"{trunc}:{qt}"] = {
            "mean": sum(ratings) / len(ratings),
            "pct_high": 100.0 * sum(x >= 5 for x in ratings) / len(ratings),
            "n": len(ratings),
        }
    io_utils.write_json(C.RESULTS_DIR / "section3_summary_fig4.json", summary)
    print("\n=== Figure 4 (mean / %>=5 by model x truncation) ===")
    for m, d in summary.items():
        for k, v in d.items():
            print(f"  {m:18s} {k:14s} mean={v['mean']:.2f} pct>=5={v['pct_high']:.1f}%")


if __name__ == "__main__":
    main()
