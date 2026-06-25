#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection.

Calibrates the detector on WildChat samples, then computes the conversation-level
emotion trajectory (Fig 14) and the layerwise pre/at/post-onset stage scores
(Fig 15) for a frustrated conversation, on both the vanilla and DPO models.

This script handles the *detection* half of Appendix I. The layer-ablation
training half is run via `train_dpo.py --layer-ablation`.

Example:
  python scripts/run_internal.py --model gemma-3-27b-it --conversation results/.../conv.json
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.models import get_client
from emotional_instability.wildchat import load_wildchat_prompts
from emotional_instability.internal.logit_emotion import (
    EmotionDetector, conversation_trajectory, layerwise_stages)
from emotional_instability.internal.ekman import build_emotion_token_dictionary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--text", default=None,
                    help="Path to a text file with the rendered conversation to analyse.")
    ap.add_argument("--onset-token-index", type=int, default=None,
                    help="Token index of emotion onset, for the layerwise stage plot.")
    ap.add_argument("--n-calibration", type=int, default=config.INTERNAL.zscore_calibration_samples)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    client = get_client(args.model)
    out_dir = os.path.join(config.RESULTS_DIR, "internal", args.model)
    io_utils.ensure_dir(out_dir)

    # Emotion-token dictionary coverage report.
    emo_tokens = build_emotion_token_dictionary(client.tokenizer)
    coverage = {e: len(ids) for e, ids in emo_tokens.items()}
    coverage["total"] = sum(coverage.values())
    io_utils.write_json(os.path.join(out_dir, "emotion_token_coverage.json"), coverage)
    print("Emotion-token coverage:", coverage)

    detector = EmotionDetector(client)

    # Calibrate on WildChat first-turn prompts (proxy for "500 WildChat samples").
    wc_prompts, _ = load_wildchat_prompts(n_prompts=args.n_calibration, seed=args.seed)
    detector.calibrate(wc_prompts, seed=args.seed)

    if args.text:
        with open(args.text) as f:
            convo = f.read()
        traj = conversation_trajectory(detector, convo)
        io_utils.write_json(os.path.join(out_dir, "trajectory.json"),
                            {e: v.tolist() for e, v in traj.items()})
        if args.onset_token_index is not None:
            stages = layerwise_stages(detector, convo, args.onset_token_index)
            io_utils.write_json(os.path.join(out_dir, "layerwise_stages.json"), stages)
        print("Wrote trajectory / stage emotion scores to", out_dir)
    else:
        print("No --text conversation provided; calibration + coverage only.")


if __name__ == "__main__":
    main()
