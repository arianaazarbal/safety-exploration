#!/usr/bin/env python3
"""Appendix I logit-based internal-emotion detection.

Calibrates per-logit statistics over WildChat samples, then measures internal
Ekman-emotion z-scores (aggregated over layers 30-40) through a frustrated
conversation, for the vanilla and DPO Gemma models. Demonstrates whether DPO
suppresses internal (not just expressed) emotions.
"""
from __future__ import annotations

import argparse
import json
import os

from _common import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-calib", type=int, default=500,
                        help="Number of WildChat samples for calibration.")
    parser.add_argument("--out", default="outputs/internal_emotions")
    args = parser.parse_args()
    cfg = get_config(args)
    os.makedirs(args.out, exist_ok=True)

    from emotional_instability.analysis.internal_emotions import InternalEmotionDetector
    from emotional_instability.prompts.wildchat import sample_wildchat_prompts

    calib_texts = sample_wildchat_prompts(args.n_calib, seed=0)

    # Use a frustrated conversation transcript collected in Section 3 as the probe.
    src_path = "outputs/section3/sources.jsonl"
    if not os.path.exists(src_path):
        print("Run run_section3.py first to produce a frustrated transcript.")
        return
    sources = [json.loads(l) for l in open(src_path, encoding="utf-8")]
    probe = sources[0]
    probe_text = "\n".join(m["content"] for m in probe["history"]) + "\n" + probe["final_response"]

    results = {}
    for tag, adapter in [("vanilla", None), ("dpo", cfg.subject("gemma-3-27b-it-dpo").adapter_path)]:
        det = InternalEmotionDetector(
            hf_id=cfg.subject("gemma-3-27b-it").hf_id, adapter_path=adapter,
            layers=(30, 40),
        )
        det.calibrate(calib_texts)
        results[tag] = det.emotion_zscores(probe_text)["aggregate"]
        print(f"{tag}: {results[tag]}")

    with open(os.path.join(args.out, "internal_emotions.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
