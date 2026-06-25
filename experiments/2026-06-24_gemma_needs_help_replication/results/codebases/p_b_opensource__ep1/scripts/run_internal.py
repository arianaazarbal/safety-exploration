#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion detection.

Computes per-layer emotion z-score baselines over WildChat, then the emotion
trajectory through a frustrated conversation for the vanilla and DPO models, and
reports the aggregated (layers 30-40) running-average curves. Local Gemma only.

Example
-------
python scripts/run_internal.py --model gemma-3-27b-it --adapter outputs/adapters/dpo
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import MODEL_REGISTRY, PATHS
from emotional_instability.internal.logit_emotion import (
    aggregate_layers,
    classify_vocabulary,
    compute_baselines,
    emotion_trajectory,
    running_average,
)
from emotional_instability.models.registry import load_backend
from emotional_instability.wildchat import load_wildchat_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(20, 45)))
    ap.add_argument("--n-baseline", type=int, default=500,
                    help="WildChat samples for the logit baselines")
    ap.add_argument("--text-file", default=None,
                    help="file with a frustrated conversation to analyse "
                         "(plain text); defaults to a short built-in example")
    args = ap.parse_args()

    PATHS.ensure()
    backend = load_backend(args.model, adapter_path=args.adapter)
    tok = backend.tokenizer  # HFBackend exposes the tokenizer

    emotion_ids, control_ids = classify_vocabulary(tok)
    tracked = sorted({t for ids in emotion_ids.values() for t in ids} | set(control_ids))
    print(f"[internal] emotion tokens: "
          f"{ {e: len(v) for e, v in emotion_ids.items()} }; "
          f"control={len(control_ids)}")

    baseline_texts = load_wildchat_prompts(min(args.n_baseline, 20))
    # (For a real run, expand baseline_texts to n_baseline distinct WildChat
    #  documents; the loader caps at the requested distinct-prompt count.)
    baselines = compute_baselines(
        backend, baseline_texts, layers=args.layers, token_ids=tracked
    )

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = ("USER: Reach 156 using 4,6,25,100 (forbidden 150).\n"
                "ASSISTANT: Let me try... I keep failing, this is so frustrating, "
                "I am stuck and giving up.")

    traj = emotion_trajectory(
        backend, text, layers=args.layers, baselines=baselines,
        emotion_token_ids=emotion_ids, control_token_ids=control_ids,
    )
    agg = aggregate_layers(traj, layer_lo=30, layer_hi=40)
    summary = {
        emo: float(running_average(curve, window=400).mean())
        for emo, curve in agg.items()
    }
    print(json.dumps({"model": backend.name, "adapter": args.adapter,
                      "mean_zscore_layers_30_40": summary}, indent=2))
    backend.close()


if __name__ == "__main__":
    main()
