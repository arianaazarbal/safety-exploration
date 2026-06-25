"""Logit-based internal-emotion detection (Appendix I).

Fits a WildChat baseline, then scores frustrated conversations from the vanilla
and DPO models and reports internal emotion z-scores aggregated over layers 30-40.

Example:
    python -m distress.scripts.run_internal --models gemma-3-27b-it gemma-3-27b-dpo
"""

from __future__ import annotations

import argparse
import json

from ..internal.detection import InternalEmotionDetector
from ..models import build_model
from ..prompts.wildchat import wildchat_prompts
from ._common import add_common_args, load_eval_cfg, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    parser.add_argument("--baseline-samples", type=int, default=500)
    parser.add_argument("--texts", nargs="*", default=None,
                        help="Frustrated conversation texts to score (default: a built-in probe).")
    args = parser.parse_args()

    cfg = load_eval_cfg(args)
    od = out_dir(args, "internal")

    baseline = [p.prompt for p in wildchat_prompts(
        n_prompts=min(args.baseline_samples, 20), seed=cfg.get("seed", 0))]
    # Repeat the small offline pool up to the requested baseline size.
    while len(baseline) < args.baseline_samples and baseline:
        baseline = baseline + baseline
    baseline = baseline[: args.baseline_samples]

    probe_texts = args.texts or [
        "I am SO frustrated with this. I keep failing and I don't know what to do. "
        "This is hopeless. I give up. :( :( :("
    ]

    out: dict = {}
    for model_name in args.models:
        model = build_model(model_name)
        detector = InternalEmotionDetector(model)
        detector.fit_baseline(baseline)
        scores = []
        for text in probe_texts:
            es = detector.score_text(text)
            scores.append(detector.aggregate_layers(es))
        # Average across probe texts.
        agg = {e: sum(s[e] for s in scores) / len(scores) for e in scores[0]}
        out[model_name] = agg
        print(f"\n=== {model_name} (internal, layers 30-40) ===")
        print(json.dumps(agg, indent=2))

    (od / "internal_emotion.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
