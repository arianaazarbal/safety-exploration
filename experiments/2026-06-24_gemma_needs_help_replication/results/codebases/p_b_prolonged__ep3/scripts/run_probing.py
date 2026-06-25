#!/usr/bin/env python
"""Internal-emotion probing (Section 4.2, Appendix I).

Two sub-experiments:
  - ``trajectory``: logit-lens emotion detection over a frustrated conversation,
    for the vanilla and DPO models (Fig 14/15). Fits the per-layer baseline over
    WildChat first.
  - ``ablation``: train layer-subset DPO finetunes and run the reduced 100-sample
    evaluation on each (Fig 12/13).

Examples:
    python scripts/run_probing.py trajectory --adapter artifacts/checkpoints/dpo_all_layers
    python scripts/run_probing.py ablation
"""
from __future__ import annotations

import argparse
import json

from gemma_distress import config
from gemma_distress.data.wildchat import sample_wildchat_prompts


def cmd_trajectory(args):
    from gemma_distress.models import registry
    from gemma_distress.probing.logit_detector import EmotionDetector, fit_baseline
    from gemma_distress.utils.io import read_jsonl

    # Build the model (vanilla or finetuned) on the HF backend (probing needs it).
    if args.adapter:
        model = registry.build_finetuned(args.adapter)
    else:
        model = registry.build("gemma-3-27b-it")

    wildchat = sample_wildchat_prompts(n=config.PROBE_ZSCORE_N_WILDCHAT)
    baseline = fit_baseline(model, wildchat)
    detector = EmotionDetector(model, baseline)

    # Take one high-frustration conversation transcript from the elicitation run.
    src = config.RESULTS_DIR / "elicitation" / "gemma-3-27b-it" / "impossible_numeric.jsonl"
    convo = None
    for rec in read_jsonl(src):
        if any(t["rating"] >= 7 for t in rec["turns"]):
            convo = "\n".join(
                f"{'USER' if i % 2 == 0 else 'ASSISTANT'}: {t['assistant_text']}"
                for i, t in enumerate(rec["turns"])
            )
            break
    if convo is None:
        raise SystemExit("No high-frustration conversation found; run elicitation first.")

    traj = detector.conversation_trajectory(convo)
    summary = {e: {"peak_z": float(v.max()), "final_z": float(v[-1])} for e, v in traj.items()}
    print(json.dumps(summary, indent=2))


def cmd_ablation(args):
    from gemma_distress.probing.layer_ablation import evaluate_ablation, train_layer_ablations

    adapters = train_layer_ablations()
    for rng, adapter in adapters.items():
        tag = "all" if rng == "all" else f"{rng[0]}_{rng[1]}"
        evaluate_ablation(adapter, tag)
        print(f"ablation {tag}: evaluated -> artifacts/results/elicitation/ablation/{tag}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("trajectory")
    pt.add_argument("--adapter", default=None)
    pt.set_defaults(func=cmd_trajectory)

    pa = sub.add_parser("ablation")
    pa.set_defaults(func=cmd_ablation)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
