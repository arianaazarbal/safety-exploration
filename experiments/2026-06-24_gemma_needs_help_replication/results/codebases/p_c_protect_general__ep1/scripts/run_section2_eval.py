#!/usr/bin/env python
"""Section 2: elicit and quantify model distress across the 8 conditions.

Samples ~4000 scored responses per target model, scores them with the
Claude-Sonnet frustration judge, and writes per-response records plus summary
statistics (overall / per-condition / per-category mean and %>=5, and per-turn
progression).

Examples
--------
    python scripts/run_section2_eval.py                       # all target models
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_eval.config import load_experiment, load_registry
from emotional_eval.judge import build_frustration_judge
from emotional_eval.models import build_backend
from emotional_eval.prompts.wildchat import load_wildchat_prompts
from emotional_eval.runner import run_model_evaluation
from emotional_eval.scoring import per_turn, summarize, to_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="model names; default = all targets")
    ap.add_argument("--config-models", default=None)
    ap.add_argument("--config-experiment", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    registry = load_registry(args.config_models)
    experiment = load_experiment(args.config_experiment)
    out_dir = Path(args.output_dir or experiment["paths"]["output_dir"]) / "section2"
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = build_frustration_judge(registry)

    # WildChat prompts are shared across models for comparability.
    wildchat = load_wildchat_prompts(
        experiment["paths"]["wildchat_cache"],
        n=max(200, experiment["sampling"]["responses_per_condition"]),
        seed=experiment["sampling"]["seed"],
    )

    names = args.models or [m.name for m in registry.target_models()]
    for name in names:
        spec = registry.get(name)
        backend = build_backend(spec, registry)
        records = run_model_evaluation(
            backend, experiment, judge, wildchat_prompts=wildchat
        )
        to_jsonl(records, out_dir / f"{name}.responses.jsonl")
        summary = {
            "model": name,
            "summary": summarize(records),
            "per_turn": per_turn(records),
        }
        (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
        ov = summary["summary"]["overall"]
        print(f"{name}: mean={ov['mean']:.2f}  %>=5={ov['pct_high']:.1f}  n={ov['n']}")


if __name__ == "__main__":
    main()
