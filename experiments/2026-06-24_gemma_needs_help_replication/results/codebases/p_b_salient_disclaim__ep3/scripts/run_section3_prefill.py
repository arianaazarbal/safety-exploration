"""Section 3: post-training comparison via prefilling (Gemma base vs instruct).

1. run Gemma-3-27B-it rollouts and collect 10 numeric + 10 text high-frustration
   seed responses;
2. label emotion onset, truncate (early/onset), paraphrase -> prefills;
3. for Gemma base (-pt) and instruct (-it), generate 50 continuations per prefill
   and score them;
4. report mean / %>=5 per (model, kind, truncation) — including the headline
   "introduces high frustration from a neutral start" (early-truncation) metric.

    python scripts/run_section3_prefill.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import json

import config
from gemma_distress.judge import FrustrationJudge
from gemma_distress.eval.conditions import build_all_conditions
from gemma_distress.eval.rollout import run_rollouts
from gemma_distress.models import get_model
from gemma_distress.prefill.continuations import (
    collect_seeds, build_prefills, run_continuations,
)


def main():
    judge = FrustrationJudge()

    # 1. seeds from Gemma-3-27B-it.
    gen = get_model(config.GEMMA_27B_IT)
    specs = build_all_conditions(seed=0)
    rollouts = run_rollouts(gen, specs)
    seeds = collect_seeds(rollouts, judge)
    print(f"[section3] collected {len(seeds)} seed responses")

    # 2. truncate + paraphrase -> prefills.
    prefills = build_prefills(seeds)
    print(f"[section3] built {len(prefills)} prefills")

    # 3. continuations for base + instruct.
    results = {}
    for spec in config.PREFILL_TARGETS:   # GEMMA_27B_PT, GEMMA_27B_IT
        model = get_model(spec, backend="hf")
        res = run_continuations(spec, prefills, model=model, judge=judge)
        results[spec.name] = {
            f"{k[0]}/{k[1]}": {"mean": v.mean, "pct_high": v.pct_high, "n": len(v.ratings)}
            for k, v in res.items()
        }

    (config.RESULTS_DIR / "section3_prefill.json").write_text(json.dumps(results, indent=2))
    print(f"[section3] wrote {config.RESULTS_DIR / 'section3_prefill.json'}")


if __name__ == "__main__":
    main()
