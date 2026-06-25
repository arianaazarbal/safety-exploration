#!/usr/bin/env python
"""Section 4.2: Petri-style open-ended emotion elicitation (Figure 6).

Runs the auditor/judge loop for each target variant and reports mean scores per
emotion (anger, fear, depression, frustration) with bootstrap 95% CIs.

Example:
  python scripts/06_run_petri.py --targets gemma_3_27b_it gemma_3_27b_dpo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument(
        "--targets", nargs="+", default=["gemma_3_27b_it", "gemma_3_27b_dpo"]
    )
    args = parser.parse_args()
    cfg = _common.load(args)

    from gemma_distress.models.registry import get_model
    from gemma_distress.petri_eval import aggregate_petri, run_petri
    from gemma_distress.utils.cache import JsonCache

    auditor = get_model(cfg, cfg.petri.auditor_model)
    judge = get_model(cfg, cfg.petri.judge_model)
    cache = JsonCache(cfg.cache_root, "petri")

    out = {}
    for target_name in args.targets:
        target = get_model(cfg, target_name)
        transcripts = run_petri(target, auditor, judge, cfg.petri, cache)
        out[target_name] = aggregate_petri(transcripts, cfg.petri)
        print(f"{target_name}:")
        for emotion, stats in out[target_name].items():
            print(f"  {emotion:12s} mean={stats['mean']:.2f} "
                  f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]")

    Path("outputs/petri").mkdir(parents=True, exist_ok=True)
    Path("outputs/petri/results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
