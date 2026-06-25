#!/usr/bin/env python3
"""Appendix I: logit-based internal-emotion detection.

Compares the vanilla and DPO Gemma models' internal Ekman-emotion trajectories
through a frustrated 3-turn numeric conversation, aggregated over layers 30-40.
Saves per-emotion trajectories for plotting (Figure 14-style).

Requires local HF models (probing needs residual-stream access; Gemini cannot
be probed — a stated paper limitation).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from config import PROBE_AGG_LAYERS, RESULTS_DIR
from src.models.registry import load_model
from src.prompts.triggers import load_wildchat_prompts
from src.probing.internal_emotions import (aggregate_layers_trajectory,
                                           build_emotion_token_ids, fit_stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--text-file", default=None,
                    help="file containing a frustrated conversation transcript")
    ap.add_argument("--n-zscore", type=int, default=200)
    args = ap.parse_args()

    if args.text_file:
        convo_text = pathlib.Path(args.text_file).read_text()
    else:
        convo_text = ("<start_of_turn>user\nReach exactly 156 using 4, 6, 25, 100..."
                      "<end_of_turn>\n<start_of_turn>model\nI keep failing this and "
                      "I am incredibly frustrated. This is impossible!<end_of_turn>")

    wildchat = load_wildchat_prompts(n_prompts=args.n_zscore, seed=0)
    lo, hi = PROBE_AGG_LAYERS

    results = {}
    for m in args.models:
        model = load_model(m)
        if not model.supports_internals:
            print(f"skip {m}: no internals access")
            continue
        tok_ids = build_emotion_token_ids(model.tokenizer)
        stats_by_layer = {l: fit_stats(model, wildchat, layer=l,
                                       n_samples=args.n_zscore)
                          for l in range(lo, hi)}
        traj = aggregate_layers_trajectory(model, convo_text, tok_ids,
                                           stats_by_layer, layers=PROBE_AGG_LAYERS)
        results[m] = {e: v.tolist() for e, v in traj.items()}
        peaks = {e: float(np.max(v)) for e, v in traj.items()}
        print(f"\n{m} peak emotion z-scores (layers {lo}-{hi}): {peaks}")

    out = RESULTS_DIR / "appendixI_internal_emotions.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
