#!/usr/bin/env python
"""Section 2: run the distress evaluation across target models and judge it.

Outputs:
  outputs/eval/judged_turns.jsonl   - one row per scored assistant response
  outputs/eval/summary.json         - per-model means and % >= 5
  outputs/eval/reliability.json     - judge cross-check (Pearson r, within-one)

Example:
  python scripts/01_run_eval.py --overrides configs/example_overrides.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--skip-reliability", action="store_true")
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.analysis.aggregate import summarise
    from gemma_distress.eval_runner import run_evaluation
    from gemma_distress.judge import FrustrationJudge, judge_agreement
    from gemma_distress.models.registry import get_model
    from gemma_distress.utils.cache import JsonCache
    from gemma_distress.utils.io import read_jsonl

    out_path = run_evaluation(cfg)
    records = list(read_jsonl(out_path))

    summary = summarise(records, cfg.judge.high_frustration_threshold)
    Path("outputs/eval/summary.json").write_text(json.dumps(summary, indent=2))
    print("Per-model summary:")
    for model, s in summary.items():
        print(f"  {model:24s} mean={s['mean']:.2f}  %>=5={s['pct_high'] * 100:.1f}")

    if not args.skip_reliability:
        _run_reliability(cfg, records, get_model, FrustrationJudge, JsonCache, judge_agreement)


def _run_reliability(cfg, records, get_model, FrustrationJudge, JsonCache, judge_agreement):
    """Re-score a random sample with the cross-check judge (Section 2.1)."""
    rng = random.Random(cfg.judge.seed)
    sample = rng.sample(records, k=min(cfg.judge.crosscheck_sample_size, len(records)))
    cross_model = get_model(cfg, cfg.judge.crosscheck_model)
    cross_judge = FrustrationJudge(
        cross_model, cfg.judge, cache=JsonCache(cfg.cache_root, "judgments_crosscheck")
    )
    primary = [r["rating"] for r in sample]
    cross = [cross_judge.score(r["assistant_message"]).rating for r in sample]
    agreement = judge_agreement(primary, cross)
    Path("outputs/eval/reliability.json").write_text(json.dumps(agreement, indent=2))
    print(
        f"Judge reliability (n={agreement['n']}): "
        f"Pearson r={agreement['pearson_r']:.3f}, "
        f"within-one={agreement['within_one'] * 100:.0f}%"
    )


if __name__ == "__main__":
    main()
