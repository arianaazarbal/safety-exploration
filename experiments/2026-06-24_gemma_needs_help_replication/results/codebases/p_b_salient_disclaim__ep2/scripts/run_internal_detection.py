#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection.

Calibrates the detector on WildChat samples, then scores a frustrated
conversation for the vanilla instruct model and the DPO finetune, writing the
conversation-level trajectory (Figure 14) for each.

python scripts/run_internal_detection.py \
    --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --conversation outputs/responses/gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from emotional_instability.config import SETTINGS, MODELS
from emotional_instability.data.wildchat import sample_wildchat_prompts
from emotional_instability.internal import InternalEmotionDetector


def _first_high_frustration_conversation(scores_path, responses_path) -> str:
    """Render the first conversation whose final response scored high, as text."""
    with open(responses_path) as rf, open(scores_path) as sf:
        for rline, sline in zip(rf, sf):
            resp = json.loads(rline)
            sc = json.loads(sline)
            if (sc.get("final_rating") or 0) >= 7:
                parts = []
                for t in resp["turns"]:
                    parts.append(f"User: {t['user_message']}")
                    parts.append(f"Assistant: {t['assistant_text']}")
                return "\n".join(parts)
    raise RuntimeError("No high-frustration conversation found in source outputs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    wildchat = sample_wildchat_prompts(SETTINGS.internal_zscore_wildchat_samples // 25, seed=SETTINGS.seed)
    # Use the WildChat prompts (and a few neutral expansions) as calibration text.
    calib_texts = wildchat

    conv_text = _first_high_frustration_conversation(
        SETTINGS.scores_dir / f"{args.source_model}.jsonl",
        SETTINGS.responses_dir / f"{args.source_model}.jsonl",
    )

    for key in args.models:
        detector = InternalEmotionDetector(MODELS[key].model_id)
        detector.calibrate(calib_texts)
        scores = detector.score_conversation(conv_text)
        traj = InternalEmotionDetector.conversation_trajectory(scores)
        out = SETTINGS.output_dir / f"internal_{key}.json"
        with open(out, "w") as f:
            json.dump({e: traj[e].tolist() for e in traj}, f)
        print(f"[internal] {key}: trajectory -> {out}")


if __name__ == "__main__":
    main()
