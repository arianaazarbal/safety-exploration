#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill comparison (Gemma 27B).

Mines high-frustration sources from the Section 2 eval records, builds early/
onset prefills (paraphrased), and measures emotion in continuations from the
base and instruct models.

Prerequisite: run scripts/01_run_eval.py first (needs judged_turns.jsonl with
gemma_3_27b_it records).

Example:
  python scripts/02_run_prefill.py --base gemma_3_27b_pt --instruct gemma_3_27b_it
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--base", default="gemma_3_27b_pt")
    parser.add_argument("--instruct", default="gemma_3_27b_it")
    parser.add_argument("--eval-records", default="outputs/eval/judged_turns.jsonl")
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.judge import FrustrationJudge
    from gemma_distress.models.registry import get_model
    from gemma_distress.prefill import (
        build_prefills,
        mine_sources,
        run_prefill_continuations,
    )
    from gemma_distress.utils.cache import JsonCache
    from gemma_distress.utils.io import read_jsonl

    records = [
        r for r in read_jsonl(args.eval_records) if r["model_name"] == args.instruct
    ]
    sources = mine_sources(records, cfg.prefill)
    print(f"Mined {len(sources)} high-frustration source responses")

    judge = FrustrationJudge(
        get_model(cfg, cfg.judge.judge_model),
        cfg.judge,
        cache=JsonCache(cfg.cache_root, "judgments"),
    )
    labeller = get_model(cfg, "onset_labeller")
    instruct = get_model(cfg, args.instruct)

    prefills = build_prefills(sources, instruct, labeller, labeller, cfg.prefill)
    print(f"Built {len(prefills)} prefills (early + onset)")

    prefill_cache = JsonCache(cfg.cache_root, "prefill")
    results_by_model: dict[str, list] = {}
    for model_name in (args.base, args.instruct):
        model = get_model(cfg, model_name)
        results = run_prefill_continuations(
            model, prefills, judge, cfg.prefill, prefill_cache
        )
        results_by_model[model_name] = results

    # Aggregate: mean and % >= 5 by (model, question_type, condition).
    out = defaultdict(dict)
    for model_name, results in results_by_model.items():
        groups = defaultdict(list)
        for r in results:
            groups[(r.question_type, r.condition)].append(r.rating)
        for (qt, cond), ratings in groups.items():
            mean = sum(ratings) / len(ratings)
            pct_high = sum(x >= 5 for x in ratings) / len(ratings)
            out[model_name][f"{qt}/{cond}"] = {
                "mean": mean,
                "pct_high": pct_high,
                "n": len(ratings),
            }

    Path("outputs/prefill").mkdir(parents=True, exist_ok=True)
    Path("outputs/prefill/results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
