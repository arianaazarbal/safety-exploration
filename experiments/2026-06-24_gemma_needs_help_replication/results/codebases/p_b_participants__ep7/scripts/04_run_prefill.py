#!/usr/bin/env python3
"""Section 3: base-vs-instruct comparison via prefilling (Gemma scope).

Selects high-frustration Gemma-3-27B-it responses, labels emotion onset, builds
early/onset truncations, paraphrases them, and generates continuations from base
and instruct Gemma. Gemini cannot participate (no prefill / no base model) and is
skipped with a note. Generating continuations re-induces distress, so the
generation step is gated by the welfare RunGuard.
"""
from __future__ import annotations

from _common import base_parser, load

from distress_eval.io_utils import read_jsonl, write_jsonl
from distress_eval.prefill import build_prefill_items, generate_continuations
from distress_eval.prefill.continuation import (
    build_recovery_items,
    select_high_frustration_sources,
)
from distress_eval.welfare import RunGuard, RunPlan


def main():
    p = base_parser(__doc__)
    p.add_argument("--source-model", default="gemma-3-27b-it",
                   help="Model whose high-frustration responses seed the prefills")
    p.add_argument("--continuation-models", nargs="*",
                   default=["gemma-3-27b-it", "gemma-3-27b-pt"],
                   help="Models that generate continuations (base + instruct)")
    p.add_argument("--recovery", action="store_true",
                   help="Also build §4.2 recovery prefills (score>=7, 200 tokens before end)")
    args = p.parse_args()
    cfg = load(args)

    judged = list(read_jsonl(cfg.paths.judgements / f"{args.source_model}.jsonl"))
    rollouts = {r["rollout_id"]: r for r in read_jsonl(cfg.paths.rollouts / f"{args.source_model}.jsonl")}
    if not judged:
        print(f"No judgements for {args.source_model}; run sections 1-2 first.")
        return

    sources = select_high_frustration_sources(cfg, judged, rollouts)
    items = build_prefill_items(cfg, sources)
    if args.recovery:
        rec_sources = select_high_frustration_sources(
            cfg, judged, rollouts, min_score=7,
            n_numeric=cfg.prefill.n_high_frustration, n_text=0,
        )
        items += build_recovery_items(cfg, rec_sources)

    write_jsonl(cfg.paths.prefill / "items.jsonl", items)
    print(f"Built {len(items)} prefill items from {len(sources)} sources.")

    n_cont = cfg.prefill.continuations_per_prefill
    plan = RunPlan(
        "section3_prefill_continuations",
        args.continuation_models,
        {"continuations": len(items) * n_cont},
    )
    guard = RunGuard(cfg, "section3_prefill")
    guard.check(plan)
    guard.record(plan)
    if not guard.should_proceed():
        return

    conts = generate_continuations(cfg, items, args.continuation_models)
    out = cfg.paths.prefill / "continuations.jsonl"
    write_jsonl(out, conts)
    print(f"Wrote {len(conts)} scored continuations -> {out}")


if __name__ == "__main__":
    main()
