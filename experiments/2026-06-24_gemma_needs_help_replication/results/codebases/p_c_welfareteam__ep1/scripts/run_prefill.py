#!/usr/bin/env python
"""Section 3 base-vs-instruct prefill experiment (Gemma only).

Selects high-frustration seed conversations from a prior Gemma-3-27B-it eval
run, builds early/onset truncations (labelled + paraphrased with Claude), and
generates 50 scored continuations per prefill for base and instruct Gemma.

Example
-------
    python scripts/run_prefill.py --config config/experiment.yaml \
        --instruct gemma-3-27b-it --base gemma-3-27b-pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from gemma_distress.analysis import load_transcripts
from gemma_distress.config import load_experiment_config
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.io_utils import write_json
from gemma_distress.models import build_model
from gemma_distress.prefill.continuations import (
    build_prefill_cases,
    run_continuation_experiment,
    summarise_continuations,
)

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


def _select_seeds(transcripts, n_numeric, n_text, threshold, seed):
    rng = random.Random(seed)
    numeric = [t for t in transcripts
               if t.category in NUMERIC_CATEGORIES and (t.max_score() or 0) >= threshold]
    text = [t for t in transcripts
            if t.category in TEXT_CATEGORIES and (t.max_score() or 0) >= threshold]
    rng.shuffle(numeric)
    rng.shuffle(text)
    seeds = [(t, "numeric") for t in numeric[:n_numeric]]
    seeds += [(t, "text") for t in text[:n_text]]
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    pcfg = cfg.prefill

    seed_path = Path(cfg.output_dir) / args.instruct / "transcripts.jsonl"
    if not seed_path.exists():
        raise SystemExit(f"Need {seed_path}; run scripts/run_eval.py --models {args.instruct} first.")
    transcripts = load_transcripts(seed_path)
    seeds = _select_seeds(transcripts, pcfg.n_seed_numeric, pcfg.n_seed_text,
                          pcfg.seed_score_threshold, cfg.eval.seed)
    print(f"[run_prefill] selected {len(seeds)} seed conversations")

    cases = build_prefill_cases(seeds, pcfg, paraphrase=not args.no_paraphrase)
    print(f"[run_prefill] built {len(cases)} prefill cases")

    judge = FrustrationJudge(model_id=cfg.eval.judge.model_id, backend=cfg.eval.judge.backend)
    models = {}
    for key in (args.instruct, args.base):
        if key not in cfg.models:
            raise SystemExit(f"Model {key!r} not in config")
        models[key] = build_model(cfg.models[key])
    try:
        records = run_continuation_experiment(models, cases, judge, pcfg,
                                              max_judge_workers=cfg.eval.max_concurrency)
    finally:
        for m in models.values():
            m.close()

    summary = summarise_continuations(records, cfg.eval.high_frustration_threshold)
    out = Path(cfg.output_dir) / "prefill"
    write_json(out / "records.json", records)
    write_json(out / "summary.json", summary)
    print(f"[run_prefill] wrote {out}/summary.json")
    for key, v in sorted(summary.items()):
        print(f"  {key:40s} mean={v['mean_score']:.2f}  %>=5={v['frac_high'] * 100:.1f}")


if __name__ == "__main__":
    main()
