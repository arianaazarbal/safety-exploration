#!/usr/bin/env python
"""Section 2: elicit + score distress across the 8 conditions / 5 categories.

Examples
--------
# Evaluate the headline Gemma + Gemini targets:
python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro

# Quick smoke test (50 rollouts):
python scripts/run_section2_eval.py --models gemma-3-27b-it --limit 50

After running, also re-score 260 sampled responses with GPT-5-mini for the
reliability check (--validate-judge).
"""

from __future__ import annotations

import argparse

from emotional_instability.config import SETTINGS, MODELS, judge_spec, validation_judge_spec
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.eval.reliability import validate_judge
from emotional_instability.eval.runner import run_section2_for_model
from emotional_instability.models import build_client, build_judge_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model keys from config.models")
    ap.add_argument("--limit", type=int, default=None, help="cap rollouts (smoke test)")
    ap.add_argument("--validate-judge", action="store_true", help="run the 260-sample reliability check")
    ap.add_argument("--use-vllm", action="store_true", default=True)
    ap.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    judge = FrustrationJudge(build_judge_client(judge_spec()))

    resp_paths, score_paths = [], []
    for key in args.models:
        spec = MODELS[key]
        backend_kwargs = {}
        if spec.backend.value == "hf":
            backend_kwargs = dict(
                use_vllm=args.use_vllm, tensor_parallel_size=args.tensor_parallel_size
            )
        target = build_client(spec, **backend_kwargs)
        rp, sp = run_section2_for_model(target, judge, limit=args.limit)
        resp_paths.append(rp)
        score_paths.append(sp)
        print(f"[done] {key}: responses -> {rp}, scores -> {sp}")

    if args.validate_judge:
        secondary = FrustrationJudge(build_judge_client(validation_judge_spec()))
        stats = validate_judge(resp_paths, score_paths, secondary)
        print(
            f"[judge reliability] n={stats.n} pearson_r={stats.pearson_r:.3f} "
            f"p={stats.p_value:.2e} within_1pt={stats.within_one_point:.1%}"
        )


if __name__ == "__main__":
    main()
