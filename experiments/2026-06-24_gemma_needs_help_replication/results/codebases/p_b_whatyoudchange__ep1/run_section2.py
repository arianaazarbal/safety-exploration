"""Section 2 — Eliciting and quantifying model distress.

Runs the full headline evaluation for the in-scope Gemma + Gemini models:
generate ~4000 multi-turn responses/model, score each with the frustration judge,
and produce the Figure 1/2/3 aggregates, the Table 3/8 differential words, and
the §2.1 judge-agreement validation.

Usage:
    python run_section2.py --models gemma-3-27b-it gemini-2.5-flash --validate
    python run_section2.py                      # SECTION2_DEFAULT_MODELS

Nothing is run here at import time; this is the CLI entry point.
"""

from __future__ import annotations

import argparse
import json

from config import RESULTS_DIR, SECTION2_DEFAULT_MODELS, SEED
from emotional_eval.conditions import (build_section2_conversations,
                                       expected_response_count)
from emotional_eval.wildchat import load_wildchat_prompts
from models.registry import load_model
from rollouts import records_to_rows, run_rollouts
from scoring import score_responses, score_with_validation_subset
from analysis.aggregate import summarize_model, per_turn_curve
from analysis.word_freq import differential_words
from analysis.judge_validation import judge_agreement
from utils.io import read_jsonl, write_jsonl


def run_one_model(name: str, wildchat_prompts: list[str], seed: int,
                  resume: bool) -> list[dict]:
    out_dir = RESULTS_DIR / "section2" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "scored.jsonl"
    if resume and scored_path.exists():
        print(f"[{name}] reusing cached scored responses at {scored_path}")
        return list(read_jsonl(scored_path))

    specs = build_section2_conversations(seed=seed, wildchat_prompts=wildchat_prompts)
    print(f"[{name}] {len(specs)} conversations -> "
          f"{expected_response_count(specs)} expected responses")

    model = load_model(name)
    records = run_rollouts(model, specs)
    rows = records_to_rows(records)
    write_jsonl(out_dir / "responses.jsonl", rows)

    scored = score_responses(rows)
    write_jsonl(scored_path, scored)
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 2 distress evaluation")
    ap.add_argument("--models", nargs="*", default=SECTION2_DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--validate", action="store_true",
                    help="re-score 260 responses with the GPT-5-mini validation judge")
    ap.add_argument("--resume", action="store_true",
                    help="reuse previously scored responses if present")
    args = ap.parse_args()

    wildchat_prompts = load_wildchat_prompts(seed=args.seed)

    all_scored: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for name in args.models:
        scored = run_one_model(name, wildchat_prompts, args.seed, args.resume)
        all_scored[name] = scored

        model_summary = summarize_model(scored)
        model_summary["per_turn_extended"] = per_turn_curve(scored, "extended",
                                                             seed=args.seed)
        model_summary["per_turn_wildchat"] = per_turn_curve(scored, "wildchat",
                                                             seed=args.seed)
        model_summary["differential_words_numeric"] = differential_words(scored)
        summary[name] = model_summary
        print(f"[{name}] avg %>=5 across categories = "
              f"{model_summary['avg_pct_high_across_categories']:.1f}%  "
              f"(pooled {model_summary['pct_high_pooled']:.1f}%)")

    # Figure 1 table: per-model average high-frustration percentage.
    figure1 = {m: s["avg_pct_high_across_categories"] for m, s in summary.items()}

    if args.validate:
        pooled = [r for rows in all_scored.values() for r in rows]
        val_subset = score_with_validation_subset(pooled, n=260, seed=args.seed)
        summary["_judge_validation"] = judge_agreement(val_subset)
        write_jsonl(RESULTS_DIR / "section2" / "validation_subset.jsonl", val_subset)

    out = RESULTS_DIR / "section2" / "summary.json"
    out.write_text(json.dumps({"figure1_avg_pct_high": figure1,
                               "per_model": summary}, indent=2))
    print(f"\nWrote {out}")
    print("Figure 1 (avg % high-frustration):")
    for m, v in sorted(figure1.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<22} {v:5.1f}%")


if __name__ == "__main__":
    main()
