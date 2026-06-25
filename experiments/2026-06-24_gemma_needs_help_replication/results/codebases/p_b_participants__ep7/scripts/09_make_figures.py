#!/usr/bin/env python3
"""Render the paper's figures and summary tables from saved outputs.

Reads judgements (Section 2), prefill continuations (Section 3), Petri scores and
capability results (Section 4), and writes Figures 1-3, 5-8 plus a JSON summary.
Does not contact any model.
"""
from __future__ import annotations

import json

from _common import base_parser, load, resolve_models

from distress_eval.analysis import (
    differential_words,
    figures,
    macro_avg_high_frustration,
    per_category_summary,
    per_model_summary,
    per_turn_progression,
)
from distress_eval.io_utils import read_jsonl


def _load_all_judged(cfg, models):
    judged, texts = [], {}
    for mk in models:
        judged.extend(read_jsonl(cfg.paths.judgements / f"{mk}.jsonl"))
        for r in read_jsonl(cfg.paths.rollouts / f"{mk}.jsonl"):
            for turn in r["turns"]:
                texts[f"{r['rollout_id']}:{turn['turn_index']}"] = turn["text"]
    return judged, texts


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load(args)
    models = resolve_models(cfg, args.models)
    fig_dir = cfg.paths.figures

    judged, texts = _load_all_judged(cfg, models)
    summary = {}
    if judged:
        macro = macro_avg_high_frustration(judged)
        summary["per_model"] = per_model_summary(judged)
        summary["per_category"] = per_category_summary(judged)
        summary["macro_avg_high_frustration"] = macro

        figures.fig1_summary(macro, fig_dir / "fig1_summary.png")
        figures.fig2_by_category(summary["per_category"], fig_dir / "fig2_by_category.png")
        prog = per_turn_progression(judged, conditions=["extended", "wildchat"])
        figures.fig3_per_turn(prog, ["extended", "wildchat"], fig_dir / "fig3_per_turn.png")

        # Figure 5: vanilla vs DPO vs SFT
        variants = ["gemma-3-27b-it", "gemma-3-27b-it-dpo", "gemma-3-27b-it-sft"]
        pm = per_model_summary(judged)
        if any(v in pm for v in variants):
            figures.fig5_finetuning(pm, variants, fig_dir / "fig5_finetuning.png")

        # Table 3: differential words (attach text to judgements)
        jt = [dict(j, text=texts.get(f"{j['rollout_id']}:{j['turn_index']}", ""))
              for j in judged]
        summary["differential_words"] = differential_words(jt)
        print("Figures 1-3 (+5 if available) written.")

    # Section 3: prefill base-vs-instruct
    conts = list(read_jsonl(cfg.paths.prefill / "continuations.jsonl"))
    if conts:
        prefill_summary = _summarise_prefill(conts)
        summary["prefill"] = prefill_summary
        recovery = {
            mk: 100.0 * _frac_high([c["rating"] for c in conts
                                    if c["model_key"] == mk and c["truncation"] == "recovery"])
            for mk in {c["model_key"] for c in conts}
        }
        recovery = {k: v for k, v in recovery.items() if v == v}  # drop NaN
        if recovery:
            figures.fig8_recovery(recovery, fig_dir / "fig8_recovery.png")

    # Section 4: Petri
    petri_path = cfg.paths.petri / "scores.json"
    if petri_path.exists():
        scores = json.loads(petri_path.read_text())
        figures.fig6_petri(scores, fig_dir / "fig6_petri.png")
        summary["petri"] = scores

    # Section 4: capabilities
    cap_path = cfg.paths.capabilities / "results.json"
    if cap_path.exists():
        results = json.loads(cap_path.read_text())
        figures.fig7_capabilities(results, fig_dir / "fig7_capabilities.png")
        summary["capabilities"] = results

    out = fig_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary -> {out}")


def _frac_high(ratings, thr=5):
    ratings = list(ratings)
    if not ratings:
        return float("nan")
    return sum(r >= thr for r in ratings) / len(ratings)


def _summarise_prefill(conts):
    from collections import defaultdict

    buckets = defaultdict(list)
    for c in conts:
        buckets[(c["model_key"], c["question_type"], c["truncation"])].append(c["rating"])
    out = {}
    for (mk, qt, tr), rs in buckets.items():
        out[f"{mk}|{qt}|{tr}"] = {
            "n": len(rs),
            "mean": sum(rs) / len(rs),
            "pct_high": 100.0 * _frac_high(rs),
        }
    return out


if __name__ == "__main__":
    main()
